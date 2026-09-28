package simu5g.simulations.nr.networks;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.ethernet.Eth10G;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import simu5g.common.binder.Binder;
import simu5g.common.carrierAggregation.CarrierAggregation;
import simu5g.nodes.NrUe;
import simu5g.nodes.gNodeB;
import simu5g.nodes.Upf;
import simu5g.world.radio.LteChannelControl;

network FourCell_Standalone
{
    parameters:
        int numUe = default(30);
        @display("i=block/network2;bgb=2000,2000");
    submodules:
        channelControl: LteChannelControl {
            @display("p=50,25;is=s");
        }
        routingRecorder: RoutingTableRecorder {
            @display("p=50,75;is=s");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=50,125");
        }
        binder: Binder {
            @display("p=50,175;is=s");
        }
        carrierAggregation: CarrierAggregation {
            @display("p=50,258;is=s");
        }
        server: StandardHost {
            @display("p=200,100;is=n;i=device/server");
        }
        router: Router {
            @display("p=400,100;i=device/smallrouter");
        }
        upf: Upf {
            @display("p=600,100");
        }
        iUpf: Upf {
            @display("p=600,300");
        }
        gnb0: gNodeB {
            @display("p=500,500;is=vl");
        }
        gnb1: gNodeB {
            @display("p=1500,500;is=vl");
        }
        gnb2: gNodeB {
            @display("p=500,1500;is=vl");
        }
        gnb3: gNodeB {
            @display("p=1500,1500;is=vl");
        }
        ue[numUe]: NrUe {
            @display("p=1000,1000");
        }
    connections:
        server.pppg++ <--> Eth10G <--> router.pppg++;
        router.pppg++ <--> Eth10G <--> upf.filterGate;
        upf.pppg++ <--> Eth10G <--> iUpf.pppg++;
        iUpf.pppg++ <--> Eth10G <--> gnb0.ppp;
        iUpf.pppg++ <--> Eth10G <--> gnb1.ppp;
        iUpf.pppg++ <--> Eth10G <--> gnb2.ppp;
        iUpf.pppg++ <--> Eth10G <--> gnb3.ppp;
        gnb0.x2++ <--> Eth10G <--> gnb1.x2++;
        gnb0.x2++ <--> Eth10G <--> gnb2.x2++;
        gnb0.x2++ <--> Eth10G <--> gnb3.x2++;
        gnb1.x2++ <--> Eth10G <--> gnb2.x2++;
        gnb1.x2++ <--> Eth10G <--> gnb3.x2++;
        gnb2.x2++ <--> Eth10G <--> gnb3.x2++;
}
