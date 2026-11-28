
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
struct Faces{int a,b;};
struct Suggestion{Faces faces; int side,source,tile; double score;};
struct Padded{Suggestion value; double padding;
 __device__ operator Suggestion() const {return value;}
 __device__ Padded& operator=(const Suggestion&s) {value=s;return *this;}
};
struct Warp{__device__ int thread_rank()const{return threadIdx.x;} __device__ int size()const{return 32;} __device__ void sync()const{__syncwarp();}};
void ck(cudaError_t e){if(e!=cudaSuccess){printf("%s\n",cudaGetErrorString(e));exit(2);}}
__host__ __device__ bool before(const Suggestion &a, const Suggestion &b) {
    if (a.faces.a != b.faces.a)
        return a.faces.a < b.faces.a;
    if (a.faces.b != b.faces.b)
        return a.faces.b < b.faces.b;
    if (a.score != b.score)
        return a.score > b.score;
    if (a.side != b.side)
        return a.side < b.side;
    if (a.tile != b.tile)
        return a.tile < b.tile;
    return a.source < b.source;
}

template <class Grid, class Item> __device__ void sort_proposals(Grid grid, Item *items, int count) {
    if (count < 2)
        return;
    const int lane = grid.thread_rank(), threads = grid.size();
    int size = 1;
    while (size < count)
        size *= 2;
    for (int i = count + lane; i < size; i += threads)
        items[i] = Suggestion{{INT_MAX, INT_MAX}, 0, 0, 0, 0};
    grid.sync();
    for (int span = 2; span <= size; span *= 2)
        for (int stride = span / 2; stride; stride /= 2) {
            for (int k = lane; k < size / 2; k += threads) {
                int i = 2 * (k / stride) * stride + k % stride, j = i + stride;
                Suggestion a = items[i], b = items[j];
                if ((i & span) ? before(a, b) : before(b, a)) {
                    items[i] = b;
                    items[j] = a;
                }
            }
            grid.sync();
        }
}

template<class Item> __global__ void bench(const Suggestion*input,Suggestion*output,int count,int repeat){
 __shared__ Item items[1024]; int lane=threadIdx.x;
 for(int rep=0;rep<repeat;++rep){
  for(int i=lane;i<count;i+=32)items[i]=input[i];__syncwarp();
  sort_proposals(Warp{},items,count);
  for(int i=lane;i<count;i+=32)output[blockIdx.x*count+i]=items[i];__syncwarp();
 }
}

__global__ void tile_sort(const Suggestion*input,Suggestion*temp,int count){
 __shared__ Suggestion items[32];int begin=32*blockIdx.x,lane=threadIdx.x;
 if(begin>=count)return;int n=min(32,count-begin),offset=blockIdx.y*count;
 if(lane<n)items[lane]=input[begin+lane];__syncwarp();
 sort_proposals(Warp{},items,n);
 if(lane<n)temp[offset+begin+lane]=items[lane];
}
__global__ void rank_merge(const Suggestion*temp,Suggestion*out,int count){
 int index=blockIdx.x*128+threadIdx.x,offset=blockIdx.y*count;
 if(index>=count)return;Suggestion a=temp[offset+index];int rank=index%32;
 for(int begin=0;begin<count;begin+=32){
  if(begin==index-index%32)continue;
  int lo=0,hi=min(32,count-begin);
  while(lo<hi){int mid=(lo+hi)/2;auto b=temp[offset+begin+mid];
   if(before(b,a)||(!before(a,b)&&begin+mid<index))lo=mid+1;else hi=mid;}
  rank+=lo;
 }
 out[offset+rank]=a;
}
int main(){
 for(int count:{1,31,32,33,512,664,1024}){
 std::vector<Suggestion> input(count),gold;unsigned rng=17;
 for(int i=0;i<count;++i){rng=1664525*rng+1013904223;input[i]={{int(rng%29),int(rng%17)},i%2,i,i%131,double(rng%257)/7.};}
 for(int i=32;i<count;i+=64)input[i]=input[0];
 gold=input;std::sort(gold.begin(),gold.end(),before);
 Suggestion *in,*temp,*out;ck(cudaMalloc(&in,count*sizeof(Suggestion)));ck(cudaMalloc(&temp,2*count*sizeof(Suggestion)));ck(cudaMalloc(&out,2*count*sizeof(Suggestion)));
 ck(cudaMemcpy(in,input.data(),count*sizeof(Suggestion),cudaMemcpyHostToDevice));
 cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
 for(int mode=0;mode<2;++mode){
 cudaGraph_t graph;cudaGraphExec_t executable;cudaStream_t stream;ck(cudaStreamCreate(&stream));
 ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
 for(int rep=0;rep<1000;++rep){
  if(mode){tile_sort<<<dim3(32,2),32,0,stream>>>(in,temp,count);rank_merge<<<dim3(8,2),128,0,stream>>>(temp,out,count);}
  else bench<Suggestion><<<2,32,0,stream>>>(in,out,count,1);
 }
 ck(cudaGetLastError());ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&executable,graph,0));
 ck(cudaGraphUpload(executable,stream));ck(cudaStreamSynchronize(stream));
 ck(cudaEventRecord(a,stream));ck(cudaGraphLaunch(executable,stream));ck(cudaEventRecord(b,stream));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));
 std::vector<Suggestion> actual(2*count);ck(cudaMemcpy(actual.data(),out,actual.size()*sizeof(Suggestion),cudaMemcpyDeviceToHost));
 for(int i=0;i<2*count;++i){auto&x=actual[i];auto&y=gold[i%count];if(x.faces.a!=y.faces.a||x.faces.b!=y.faces.b||x.score!=y.score||x.side!=y.side||x.source!=y.source||x.tile!=y.tile)return 1;}
 printf("count=%d method=%s milliseconds=%.6f correct=1\n",count,mode?"rank_merge":"shared_network",ms);
 ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));
 }
 ck(cudaFree(in));ck(cudaFree(temp));ck(cudaFree(out));ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
 }
}
