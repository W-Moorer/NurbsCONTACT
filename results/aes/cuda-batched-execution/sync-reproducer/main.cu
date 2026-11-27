#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
void ck(cudaError_t e){if(e!=cudaSuccess){printf("%s\n",cudaGetErrorString(e));exit(1);}}
__global__ void set(int *p){*p=1;}
__global__ void work(int *p){__shared__ int ready;if(!threadIdx.x)ready=*p;__syncthreads();if(!ready)return;
 auto b=cooperative_groups::this_thread_block();b.sync();for(int i=0;i<8;++i){b.sync();}if(!threadIdx.x)atomicAdd(p,1);}
__global__ void launch(cudaGraphExec_t e){cudaGraphLaunch(e,cudaStreamGraphTailLaunch);}
__global__ void repeat(cudaGraphConditionalHandle h,int *p){cudaGraphSetConditional(h,*p<6);}
__global__ void early(int*p){if(threadIdx.x)return;*p+=0;}
__global__ void enable(cudaGraphConditionalHandle h){cudaGraphSetConditional(h,1);}
template<class F,class A> cudaGraphNode_t node(cudaGraph_t g,cudaGraphNode_t prev,F f,int threads,A a){
 void*args[]={&a};cudaKernelNodeParams p{};p.func=(void*)f;p.gridDim=dim3(1);p.blockDim=dim3(threads);p.kernelParams=args;
 cudaGraphNode_t n;ck(cudaGraphAddKernelNode(&n,g,prev?&prev:nullptr,prev?1:0,&p));return n;}
int main(int argc,char**argv){int mode=argc>1?atoi(argv[1]):0;int*p;ck(cudaMalloc(&p,4));ck(cudaMemset(p,1,4));
 cudaGraph_t root,body;ck(cudaGraphCreate(&root,0));body=root;
 cudaGraphConditionalHandle loop{};
 if(mode>=2){cudaGraphConditionalHandle h;ck(cudaGraphConditionalHandleCreate(&h,root,0,0));auto a=node(root,{},enable,1,h);
 cudaGraphNodeParams c{};c.type=cudaGraphNodeTypeConditional;c.conditional.handle=h;c.conditional.type=(mode==2?cudaGraphCondTypeIf:cudaGraphCondTypeWhile);c.conditional.size=1;loop=h;
 cudaGraphNode_t n;ck(cudaGraphAddNode(&n,root,&a,1,&c));body=c.conditional.phGraph_out[0];}
 if(mode==1)ck(cudaGraphCreate(&body,0));
 auto a=mode<3?node(body,{},set,1,p):cudaGraphNode_t{};
 if(mode>=3){
 a=node(body,{},early,128,p);a=node(body,a,work,128,p);a=node(body,a,early,1,p);a=node(body,a,work,mode==4?128:32,p);
 void*args[]={&loop,&p};cudaKernelNodeParams r{};r.func=(void*)repeat;r.gridDim=dim3(1);r.blockDim=dim3(1);r.kernelParams=args;cudaGraphNode_t n;ck(cudaGraphAddKernelNode(&n,body,&a,1,&r));
 }else node(body,a,work,128,p);
 cudaGraphExec_t e,child{};if(mode==1){ck(cudaGraphInstantiate(&child,body,cudaGraphInstantiateFlagDeviceLaunch));ck(cudaGraphUpload(child,0));node(root,{},launch,1,child);}
 ck(cudaGraphInstantiate(&e,root,mode==1?cudaGraphInstantiateFlagDeviceLaunch:0));ck(cudaGraphUpload(e,0));ck(cudaDeviceSynchronize());ck(cudaGraphLaunch(e,0));ck(cudaDeviceSynchronize());
 int out;ck(cudaMemcpy(&out,p,4,cudaMemcpyDeviceToHost));printf("mode %d result %d\n",mode,out);return (mode>=3 || out==2)?0:1;}
