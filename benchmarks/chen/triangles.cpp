#include "config.h"
#include "triBezier.h"
#include "solverTD.h"

// One TDIB-E query in the published order: A(0), B(0), A(1), B(1).
// The exact labels stay with the caller. The 2024 solver only sees doubles.
int main(int argc,char** argv) {
    try {
        if (argc!=2) throw std::invalid_argument("nc_chen_triangles PARAMETER_TIME_WIDTH < query.txt");
        const std::string setting=argv[1];
        std::size_t used=0;
        const double width=std::stod(setting,&used);
        if (used!=setting.size() || !(width>0 && width<=1)) throw std::invalid_argument("invalid cell width");
        std::array<TriQuadBezier,4> patch;
        for (auto& p:patch) for (auto& v:p.ctrlp)
            if (!(std::cin>>v.x()>>v.y()>>v.z()) || !v.allFinite())
                throw std::invalid_argument("expected 24 finite control points");
        if (!(std::cin>>std::ws).eof()) throw std::invalid_argument("unexpected data after the query");
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
