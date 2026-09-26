"""
wdt_checkpoint.py -- restore a trained WirelessDT-6G model together with
the normalisation it was trained under.

A state_dict alone is not a usable model here. The network was fitted on
z-scored inputs and z-scored energy/delay/jitter labels, so applying the
weights to raw CSV values produces confident nonsense rather than an
error. The checkpoint therefore carries the feature means and standard
deviations, both label normalisers and the packet-loss bin edges, and
this module puts them back.

C4 depends on this: evaluating a counterfactual means building a modified
graph, normalising it exactly as the training data was normalised, running
a forward pass, and converting the outputs back to milliseconds and watts.
Any mismatch at either end silently changes the answer.
"""

import numpy as np
import torch

from wdt_model import WirelessDT6G, WirelessDT6GBlind


class LoadedNormalizer:
    def __init__(self, cols, mean, std):
        self.cols = cols
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.std_ = np.asarray(std, dtype=np.float32)

    def transform(self, df):
        a = df[self.cols].to_numpy(dtype=np.float32)
        return (a - self.mean_) / self.std_


class LoadedLabelNorm:
    def __init__(self, mean, std):
        self.mean_ = float(mean)
        self.std_ = float(std)

    def transform(self, y):
        return (np.asarray(y, dtype=np.float32) - self.mean_) / self.std_

    def inverse(self, y):
        if torch.is_tensor(y):
            return y * self.std_ + self.mean_
        return np.asarray(y, dtype=np.float32) * self.std_ + self.mean_


def load_checkpoint(path, device="cpu"):
    """-> (model in eval mode, dict of everything needed to use it)."""
    ck = torch.load(path, map_location=device, weights_only=False)

    Cls = WirelessDT6GBlind if ck.get("kind") == "blind" else WirelessDT6G
    model = Cls(ck["in_dims"], ck["relations"], hidden=ck["hidden"],
                rounds=ck["rounds"], num_pl_bins=ck["n_bins"],
                use_moe=ck["use_moe"], dropout=ck.get("dropout", 0.1))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    ctx = {
        "gnb_norm": LoadedNormalizer(ck["gnb_cols"],
                                     ck["gnb_norm"]["mean"], ck["gnb_norm"]["std"]),
        "ue_norm": LoadedNormalizer(ck["ue_cols"],
                                    ck["ue_norm"]["mean"], ck["ue_norm"]["std"]),
        "energy_norm": LoadedLabelNorm(**ck["energy_norm"]),
        "kpi_norm": {k: LoadedLabelNorm(**v) for k, v in ck["kpi_norm"].items()},
        "pl_bin_edges": ck["pl_bin_edges"],
        # Column lists passed through as well: anything rebuilding the input
        # pipeline needs to know which features the weights expect, and in
        # what order.
        "gnb_cols": ck["gnb_cols"],
        "ue_cols": ck["ue_cols"],
        "flags": ck["flags"],
        "relations": ck["relations"],
        "test_seed": ck["test_seed"],
        "train_seeds": ck.get("train_seeds"),
        "test_metrics": ck.get("test_metrics"),
    }
    return model, ctx


@torch.no_grad()
def predict(model, ctx, graph, device="cpu"):
    """Forward pass on one HeteroData, returned in physical units.

    energy watts, delay ms, jitter ms, packet-loss probability.
    """
    graph = graph.to(device)
    ea = {}
    for rel in graph.edge_types:
        store = graph[rel]
        if hasattr(store, "edge_attr") and store.edge_attr is not None:
            ea[rel] = store.edge_attr
    out = model(graph.x_dict, graph.edge_index_dict, ea)
    return {
        "energy_w": ctx["energy_norm"].inverse(out["energy"].squeeze(-1).cpu()),
        "delay_ms": ctx["kpi_norm"]["delay"].inverse(out["delay"].squeeze(-1).cpu()),
        "jitter_ms": ctx["kpi_norm"]["jitter"].inverse(out["jitter"].squeeze(-1).cpu()),
        "pl_prob": torch.sigmoid(out["pl_binary_logit"].cpu()),
        "pl_bin_logits": out["pl_bin_logits"].cpu(),
    }


if __name__ == "__main__":
    import sys
    m, c = load_checkpoint(sys.argv[1])
    print("relations:", c["relations"])
    print("flags:", c["flags"])
    print("held-out topology:", c["test_seed"], "| trained on", c["train_seeds"])
    if c["test_metrics"]:
        for k in ("energy", "delay", "jitter"):
            print(f"  {k:<8} R2 {c['test_metrics'][k]['R2']:.4f}")
        print(f"  packet_loss AUROC {c['test_metrics']['packet_loss']['AUROC']:.4f}")