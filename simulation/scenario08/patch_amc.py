"""
patch_amc.py -- fixes the LteAmc::existTxParams crash that blocks
8 gNB / 80 UE mobile scenarios.

    python3 patch_amc.py --check
    python3 patch_amc.py --apply     # then: build_simu5g

THE BUG (confirmed by stack trace, not guessed)
------------------------------------------------
lldb backtrace at the crash point, seed 41 p43, t=14.05s:

  #2 std::map<MacNodeId, unsigned>::at(...)   stl_map.h:576
  #3 simu5g::LteAmc::existTxParams(id=7, UL)  LteAmc.cc:495
  #4 simu5g::AmcPilotAuto::computeTxParams    AmcPilotAuto.cc:24
  #6 simu5g::LteSchedulerEnbUl::racschedule   LteSchedulerEnbUl.cc:85
  #9 simu5g::LteMacEnb::handleSelfMessage     LteMacEnb.cc:840

The offending line is:

  return (*txParams)[carrierFrequency].at(nodeIndex.at(id)).isValid();

`nodeIndex.at(id)` throws std::out_of_range when `id` is not in the
map. During handover a UE can issue a RAC request to a gNB that has no
transmission parameters recorded for it yet, and the uplink scheduler
then asks whether such parameters exist.

WHY THE FIX IS SAFE
-------------------
`existTxParams` is a predicate: its entire job is to answer "do
transmission parameters exist for this node?". For an unknown node the
correct answer is false, not an exception. The function ALREADY does
exactly this two lines above for an unknown carrier frequency:

  if (txParams->find(carrierFrequency) == txParams->end())
      return false;

So returning false for an unknown node id is the function's own
established behaviour, not a new convention. The patch adds the missing
symmetric guard, plus a bounds check on the vector index it feeds.

Behaviour for any node that IS present is bit-identical. The only
change is that a query about an absent node returns false instead of
terminating the simulation -- and callers already handle false, since
that is what they get for an absent carrier.

CRASH RATE BEFORE THE PATCH (handover-frequency dependent)
    4 gNB / 30 UE   20 of 21 runs completed
    6 gNB / 60 UE    9 of ~15
    8 gNB / 80 UE    2 of 28
A collaborator's 8 gNB scenario is unaffected only because its UEs are
stationary, so no handover occurs.

Document this in the thesis as a simulator patch, with the stack trace
as evidence. Re-simulate any dataset intended to be compared with runs
produced by the patched build.
"""

import argparse
import os
import shutil

PATH = os.path.expanduser(
    "~/simu5g-workspace/simu5g-1.4.4/src/simu5g/stack/mac/amc/LteAmc.cc")

OLD = """    std::map<MacNodeId, unsigned int>& nodeIndex = (dir == DL) ? dlNodeIndex_ : (dir == UL) ? ulNodeIndex_ : d2dNodeIndex_;
    return (*txParams)[carrierFrequency].at(nodeIndex.at(id)).isValid();"""

NEW = """    std::map<MacNodeId, unsigned int>& nodeIndex = (dir == DL) ? dlNodeIndex_ : (dir == UL) ? ulNodeIndex_ : d2dNodeIndex_;
    // Patch: nodeIndex.at(id) threw std::out_of_range when a UE issued a
    // RAC request to a gNB holding no tx parameters for it yet -- routine
    // during handover, and fatal at 8 gNB / 80 UE (2 of 28 runs survived).
    // This is a PREDICATE: "false" is the correct answer for an unknown
    // node, exactly as the function already returns false for an unknown
    // carrier frequency two lines above. Behaviour for present nodes is
    // unchanged.
    auto nodeIt = nodeIndex.find(id);
    if (nodeIt == nodeIndex.end())
        return false;
    std::vector<UserTxParams>& params = (*txParams)[carrierFrequency];
    if (nodeIt->second >= params.size())
        return false;
    return params.at(nodeIt->second).isValid();"""


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    backup = PATH + ".orig"

    if args.revert:
        if not os.path.exists(backup):
            raise SystemExit("no .orig backup found")
        shutil.copy(backup, PATH)
        print(f"reverted {PATH} from {backup}\nrun build_simu5g")
        return

    if not os.path.exists(PATH):
        raise SystemExit(f"not found: {PATH}")
    s = open(PATH).read()

    if NEW.split("\n")[1].strip() in s:
        print("already patched")
        return
    if OLD not in s:
        print("PATTERN NOT FOUND -- LteAmc.cc differs from expected.")
        print("Show these lines and the patch can be adjusted:")
        print("  sed -n '480,500p' " + PATH)
        raise SystemExit(1)

    print("found the unguarded lookup at LteAmc::existTxParams")
    if args.check:
        print("check only -- nothing written. Re-run with --apply.")
        return

    if not os.path.exists(backup):
        shutil.copy(PATH, backup)
        print(f"backup written: {backup}")
    open(PATH, "w").write(s.replace(OLD, NEW))
    print("patched. Now run:  build_simu5g")
    print("Then test the config that crashed deterministically:")
    print("  simu5g -f omnetpp_s41.ini -c Scenario08s41_p43 -u Cmdenv 2>&1 | tail -3")
    print("It previously died at t=14.051s. Reaching t=120s means the fix holds.")


if __name__ == "__main__":
    main()
