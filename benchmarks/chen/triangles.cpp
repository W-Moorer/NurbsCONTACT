#include "config.h"
#include "triBezier.h"
#include "solverTD.h"
#include "nurbscontact/tdib.hpp"

static std::vector<nc::HPoint> triangle(const TriQuadBezier& tri) {
    std::vector<nc::HPoint> result;
    for (const auto& q:tri.ctrlp) result.push_back(nc::homogeneous(nc::exact({q.x(),q.y(),q.z()})));
    return result;
}

// One TDIB-E query in the published order: A(0), B(0), A(1), B(1).
// The exact labels stay with the caller. The 2024 solver only sees doubles.
int main(int argc,char** argv) {
    try {
        const bool robust=argc==3 && std::string_view(argv[2])=="--robust";
        if (argc!=2 && !robust) throw std::invalid_argument("nc_chen_triangles PARAMETER_TIME_WIDTH [--robust] < query.txt");
        const std::string setting=argv[1];
        std::size_t used=0;
        const double width=std::stod(setting,&used);
        if (used!=setting.size() || !(width>0 && width<=1)) throw std::invalid_argument("invalid cell width");
        std::array<TriQuadBezier,4> patch;
        for (auto& p:patch) for (auto& v:p.ctrlp)
            if (!(std::cin>>v.x()>>v.y()>>v.z()) || !v.allFinite())
                throw std::invalid_argument("expected 24 finite control points");
        if (!(std::cin>>std::ws).eof()) throw std::invalid_argument("unexpected data after the query");
        if (robust) {
            const auto start=std::chrono::steady_clock::now();
            const auto layout=nc::BezierLayout::quadratic_triangle;
            const auto r=nc::tdib_envelopes({3,3,triangle(patch[0]),triangle(patch[2]),layout},
                                            {3,3,triangle(patch[1]),triangle(patch[3]),layout},width,1.,0.,100000);
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            const char* status=r.status==nc::TdibStatus::clear ? "clear" :
                               r.status==nc::TdibStatus::candidate ? "candidate" : "unresolved";
            std::cout<<std::setprecision(17)<<"{\"outcome\":\""<<status<<"\",\"nodes\":"<<r.nodes<<",\"query_s\":"<<seconds;
            if (r.status!=nc::TdibStatus::clear) std::cout<<",\"time\":["<<r.time.lo<<','<<r.time.hi<<']';
            std::cout<<"}\n";
            return 0;
        }
        for (unsigned i=0;i<2;++i) for (unsigned j=0;j<6;++j) patch[i+2].ctrlp[j]-=patch[i].ctrlp[j];
        Array2d a,b;
        using Solver=SolverTD<TriQuadBezier,TriQuadBezier,TriParamBound,TriParamBound>;
        const auto start=std::chrono::steady_clock::now();
        const double t=Solver::solveCCD(patch[0],patch[2],patch[1],patch[3],a,b,BoundingBoxType::OBB,width,1.);
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if (t!=-1 && !(t>=0 && t<=1 && a.allFinite() && b.allFinite()))
            throw std::runtime_error("invalid upstream candidate");
        std::cout<<std::setprecision(17)<<"{\"candidate\":"<<(t==-1 ? "false":"true")<<",\"query_s\":"<<seconds;
        if (t!=-1) std::cout<<",\"fraction\":"<<t<<",\"uv_a\":["<<a[0]<<','<<a[1]
                            <<"],\"uv_b\":["<<b[0]<<','<<b[1]<<']';
        std::cout<<"}\n";
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
