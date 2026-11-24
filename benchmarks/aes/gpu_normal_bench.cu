// nvcc -O3 -Isrc benchmarks/aes/gpu_normal_bench.cu -o gpu_normal_bench
// Two simultaneous contacts share the middle of three unit masses.
#include "gpu_normal_contact.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

static void check(cudaError_t error) {
    if (error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> static T* allocate(int n) {
    T* result;
    check(cudaMalloc(&result,sizeof(T)*n));
    return result;
}
int main() {
    try {
        const double step=1e-4, tolerance=1e-10;
        const double W[]={2.,-1.,-1.,2.}, k[]={1e8,1e8}, gaps[]={-.25*step,-step};
        auto w=allocate<double>(4), stiffness=allocate<double>(2), gap=allocate<double>(2);
        auto law=allocate<int>(2), status=allocate<int>(1), sweeps=allocate<int>(1);
        auto p=allocate<double>(2), residual=allocate<double>(1);
        check(cudaMemcpy(w,W,sizeof(W),cudaMemcpyHostToDevice));
        check(cudaMemcpy(stiffness,k,sizeof(k),cudaMemcpyHostToDevice));
        check(cudaMemcpy(gap,gaps,sizeof(gaps),cudaMemcpyHostToDevice));
        cudaStream_t stream;
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        cudaEvent_t start,stop;
        check(cudaEventCreate(&start));
        check(cudaEventCreate(&stop));
        cudaDeviceProp properties;
        check(cudaGetDeviceProperties(&properties,0));
        std::printf("{\"device\":\"%s\",\"cases\":[\n",properties.name);
        for(int a=0;a<3;++a) for(int b=0;b<3;++b) {
            const int laws[]={a,b};
            check(cudaMemcpy(law,laws,sizeof(laws),cudaMemcpyHostToDevice));
            auto run=[&] {
                solve_normal_rows<<<1,32,5*2*sizeof(double),stream>>>(w,gap,law,stiffness,2,step,tolerance,
                                                                  4000,p,residual,sweeps,status);
            };
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            // This measures one converged two-contact solve, amortizing dispatch.
            for(int i=0;i<16;++i) run();
            check(cudaStreamEndCapture(stream,&graph));
            check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            check(cudaGraphLaunch(exec,stream));
            check(cudaEventRecord(start,stream));
            for(int j=0;j<16;++j) check(cudaGraphLaunch(exec,stream));
            check(cudaEventRecord(stop,stream));
            check(cudaEventSynchronize(stop));
            float ms;
            check(cudaEventElapsedTime(&ms,start,stop));
            int code,n;
            double actual[2],error;
            check(cudaMemcpy(actual,p,sizeof(actual),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(&n,sweeps,sizeof(n),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(&code,status,sizeof(code),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(&error,residual,sizeof(error),cudaMemcpyDeviceToHost));
            const double aa=2+(a==penalty),bb=2+(b==penalty),det=aa*bb-1;
            const double diff=std::max(std::abs(actual[0]-(bb*.25+1)/det),std::abs(actual[1]-(aa+.25)/det));
            if(code || !(diff<1e-9 && error<=tolerance)) throw std::runtime_error("Incorrect normal impulse");
            std::printf("{\"laws\":[%d,%d],\"sweeps\":%d,\"max_impulse_error_Ns\":%.12g,"
                        "\"residual_m_s\":%.12g,\"mean_device_us\":%.6f}%s\n",
                        a,b,n,diff,error,ms*1000./256,(a==2 && b==2)?"":",");
            check(cudaGraphExecDestroy(exec));
            check(cudaGraphDestroy(graph));
        }
        std::puts("]}");
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"%s\n",e.what());
        return 1;
    }
}
