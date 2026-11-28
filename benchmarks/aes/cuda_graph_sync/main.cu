// Independent reproducer; not a replacement for checking the production graph.
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); }
}
struct State { unsigned count, iterations, bad, mask; };
template<int Mode> __global__ void work(State *state) {
    __shared__ unsigned values[128];
    unsigned lane = threadIdx.x % 32, base = threadIdx.x - lane;
    values[threadIdx.x] = threadIdx.x;
    if constexpr (Mode == 0) __syncthreads();
    if constexpr (Mode == 1) __syncwarp(state->mask);
    // Deliberately invalid: executing lanes 16-31 are absent from the mask.
    // Run this mode only under synccheck as a checker negative control.
    if constexpr (Mode == 2) __syncwarp(state->mask);
    if (values[base + (lane + 1) % 32] != base + (lane + 1) % 32)
        atomicAdd(&state->bad, 1u);
    if (!lane) atomicAdd(&state->count, 1u);
}
__global__ void begin(cudaGraphConditionalHandle handle) { cudaGraphSetConditional(handle, 1); }
__global__ void again(cudaGraphConditionalHandle handle, State *state) {
    cudaGraphSetConditional(handle, ++state->iterations < 5);
}
struct Builder {
    cudaGraph_t graph;
    cudaGraphNode_t last{};
    template<class F, class... A> void kernel(F f, int blocks, int threads, A... a) {
        void *args[] = {&a...};
        cudaKernelNodeParams p{};
        p.func = reinterpret_cast<void *>(f); p.gridDim = dim3(blocks); p.blockDim = dim3(threads);
        p.kernelParams = args;
        cudaGraphNode_t node;
        check(cudaGraphAddKernelNode(&node, graph, last ? &last : nullptr, last ? 1 : 0, &p));
        last = node;
    }
};
int main(int argc, char **argv) {
    if (argc != 4 || (std::strcmp(argv[1], "flat") && std::strcmp(argv[1], "if") && std::strcmp(argv[1], "while")) ||
        (std::strcmp(argv[2], "block") && std::strcmp(argv[2], "warp") && std::strcmp(argv[2], "invalid")) ||
        (std::atoi(argv[3]) != 32 && std::atoi(argv[3]) != 128)) {
        std::fprintf(stderr, "usage: probe flat|if|while block|warp|invalid 32|128\n"); return 2;
    }
    const bool conditional = std::strcmp(argv[1], "flat") != 0, loop = std::strcmp(argv[1], "while") == 0;
    const int threads = std::atoi(argv[3]);
    State *state;
    State initial{0, 0, 0, !std::strcmp(argv[2], "invalid") ? 0x0000ffffu : 0xffffffffu};
    check(cudaMalloc(&state, sizeof(State)));
    check(cudaMemcpy(state, &initial, sizeof(State), cudaMemcpyHostToDevice));
    cudaGraph_t root; check(cudaGraphCreate(&root, 0)); Builder body{root};
    cudaGraphConditionalHandle handle{};
    if (conditional) {
        check(cudaGraphConditionalHandleCreate(&handle, root, 0, 0));
        body.kernel(begin, 1, 1, handle);
        cudaGraphNodeParams p{}; p.type = cudaGraphNodeTypeConditional;
        p.conditional.handle = handle; p.conditional.type = loop ? cudaGraphCondTypeWhile : cudaGraphCondTypeIf;
        p.conditional.size = 1;
        cudaGraphNode_t node;
#if CUDART_VERSION >= 13000
        check(cudaGraphAddNode(&node, root, &body.last, nullptr, 1, &p));
#else
        check(cudaGraphAddNode(&node, root, &body.last, 1, &p));
#endif
        body = {p.conditional.phGraph_out[0]};
    }
    for (int i = 0; i < 2; ++i) {
        if (!std::strcmp(argv[2], "block")) body.kernel(work<0>, 2, threads, state);
        else if (!std::strcmp(argv[2], "warp")) body.kernel(work<1>, 2, threads, state);
        else body.kernel(work<2>, 2, threads, state);
    }
    if (loop) body.kernel(again, 1, 1, handle, state);
    cudaGraphExec_t executable; check(cudaGraphInstantiate(&executable, root, 0));
    check(cudaGraphUpload(executable, 0)); check(cudaDeviceSynchronize());
    check(cudaGraphLaunch(executable, 0)); check(cudaDeviceSynchronize());
    State result; check(cudaMemcpy(&result, state, sizeof(State), cudaMemcpyDeviceToHost));
    unsigned expected = (loop ? 5 : 1) * 4 * (threads / 32);
    std::printf("graph=%s sync=%s threads=%d count=%u expected=%u bad=%u\n",
                argv[1], argv[2], threads, result.count, expected, result.bad);
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(root)); check(cudaFree(state));
    return result.count != expected || result.bad ? 1 : 0;
}
