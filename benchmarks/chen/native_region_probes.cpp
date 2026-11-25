#include "../../src/cad_patch_search.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>

namespace {
using namespace nc;
using namespace nc::dynamics;
using namespace nc::cad;
using Clock=std::chrono::steady_clock;
ContactPose read_pose() {
    ContactPose pose;
    for (auto& x:pose.origin) std::cin>>x;
    for (auto& x:pose.axis) std::cin>>x;
    if (!(std::cin>>pose.angle)) throw std::runtime_error("incomplete pose");
    return pose;
}

void patch_replay(PatchSearch& search,int order) {
    while ((std::cin>>std::ws).peek()!=std::char_traits<char>::eof()) {
        const std::array poses{read_pose(),read_pose()};
        std::map<int,std::vector<CadChord>> groups;
        int count;
        if (!(std::cin>>count) || count<0) throw std::invalid_argument("invalid patch witness count");
        bool resolved=true;
        const auto start=Clock::now();
        for (int i=0;i<count;++i) {
            int label; Vec p,n; std::array<std::size_t,2> hints;
            std::cin>>label>>hints[0]>>hints[1];
            for (auto& x:p) std::cin>>x;
            for (auto& x:n) std::cin>>x;
            if (!std::cin || label<0) throw std::invalid_argument("invalid patch witness");
            auto w=search.chord(poses,p,n,hints);
            if (!w) resolved=false; else groups[label].push_back(*w);
        }
        std::vector<ContactRegion> result;
        for (const auto& [label,group]:groups) {
            auto r=search.integrate(poses,group,order);
            if (!r) {resolved=false;break;}
            result.push_back(*r);
        }
        std::cout<<"{\"query_s\":"<<std::chrono::duration<double>(Clock::now()-start).count()
                 <<",\"resolved\":"<<(resolved ? "true":"false")<<",\"patches\":[";
        if (resolved) for (std::size_t i=0;i<result.size();++i) {
            if (i) std::cout<<',';
            std::cout<<'[';
            for (auto v:{result[i].point,result[i].normal,result[i].pressure_arm}) for (double x:v) std::cout<<x<<',';
            std::cout<<result[i].depth_m<<']';
        }
        std::cout<<"]}"<<std::endl;
    }
}
}

int main(int argc,char** argv) try {
    if (argc>=3 && std::string_view(argv[1])=="--prepare-parts") {
        for (int i=2;i<argc;++i) prepare_patch_part(read_step_shape(argv[i]));
        return 0;
    }
    const bool dense=argc==6 && std::string_view(argv[5])=="--prepare-all";
    const bool prepare=dense || (argc==6 && std::string_view(argv[5])=="--prepare");
    const bool indexed=argc==6 && std::string_view(argv[5])=="--indexed-groups";
    const bool group=indexed || (argc==6 && std::string_view(argv[5])=="--groups");
    const bool patch=argc==7 && std::string_view(argv[5])=="--patch-wrench";
    if (argc!=5 && !group && !patch && !prepare) throw std::invalid_argument("nc_cad_probes A.step B.step sites_per_uv_axis reach_m [--prepare|--prepare-all|--groups|--indexed-groups|--patch-wrench ORDER]");
    const int grid=std::stoi(argv[3]);
    const double reach=std::stod(argv[4]);
    if (grid<1 || grid>128 || !(reach>0 && std::isfinite(reach))) throw std::invalid_argument("invalid probe grid or reach");
    const auto start=Clock::now();
    PatchSearch search(read_step_shape(argv[1]),read_step_shape(argv[2]),grid,reach);
    std::cerr<<"prepare_s "<<std::chrono::duration<double>(Clock::now()-start).count()<<'\n';
    std::cout<<std::setprecision(17);
    if (prepare) {
        if (dense) search.prepare_dense();
        const auto counts=search.asset_counts();
        std::cout<<"{\"tables_loaded\":"<<counts[0]<<",\"tables_built\":"<<counts[1]<<"}"<<std::endl;
        return 0;
    }
    if (patch) {
        const int order=std::stoi(argv[6]);
        if (order<0 || order==1 || order>256) throw std::invalid_argument("invalid surface quadrature order");
        patch_replay(search,order);
        std::cerr<<"skipped_integral_nodes "<<search.skipped_nodes()<<std::endl;
        return 0;
    }

    while ((std::cin>>std::ws).peek()!=std::char_traits<char>::eof()) {
        const std::array poses{read_pose(),read_pose()};
        const auto result=search.discover(poses,indexed,group);
        const auto& found=result.witnesses;
        const auto& labels=result.labels;
        std::cout<<"{\"query_s\":"<<result.query_s<<",\"broad_s\":"<<result.broad_s
                 <<",\"window_builds\":"<<result.window_builds<<",\"window_reuses\":"<<result.window_reuses<<",\"reused_paths\":"<<result.reused_paths<<",\"candidate_face_pairs\":"<<result.candidate_pairs
                 <<",\"attempts\":"<<result.attempts<<",\"tracked\":"<<result.tracked<<",\"face_tests\":"<<result.tests<<",\"witnesses\":[";
        for (std::size_t i=0;i<found.size();++i) {
            if (i) std::cout<<',';
            const auto& w=*found[i];
            std::cout<<'['<<w.exits[0].face<<','<<w.exits[1].face;
            for (double x:w.region.point) std::cout<<','<<x;
            for (double x:w.region.normal) std::cout<<','<<x;
            std::cout<<','<<w.region.depth_m<<']';
        }
        std::cout<<']';
        if (group) {
            std::cout<<",\"group_s\":"<<result.group_s<<",\"labels\":[";
            for (std::size_t i=0;i<labels.size();++i) {if (i) std::cout<<','; std::cout<<labels[i];}
            std::cout<<']';
        }
        std::cout<<"}"<<std::endl;
    }
    return 0;
} catch (const Standard_Failure& e) {std::cerr<<e.GetMessageString()<<'\n'; return 1;}
  catch (const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
