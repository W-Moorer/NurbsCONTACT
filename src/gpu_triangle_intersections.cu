// Traverse target AABBs for each source leaf, then test the finite triangles in each surviving pair.
#include <cuda_runtime.h>



namespace {
using V = double3;
__device__ V minus_v(V a,V b) {return make_double3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ double proj(V a,V b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ V cross_v(V a,V b) {
    return make_double3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
}
__device__ double min3(double a,double b,double c) {return fmin(a,fmin(b,c));}
__device__ double max3(double a,double b,double c) {return fmax(a,fmax(b,c));}
__device__ bool separates(V n,const V a[3],const V b[3]) {
    const double p=proj(n,a[0]),q=proj(n,a[1]),r=proj(n,a[2]);
    const double s=proj(n,b[0]),t=proj(n,b[1]),u=proj(n,b[2]);
    return max3(p,q,r)<min3(s,t,u) || max3(s,t,u)<min3(p,q,r);
}
__device__ bool intersect(const V first[3],const V second[3]) {
    V a[3],b[3],ea[3],eb[3];
    for (int i=0;i<3;++i) {
        a[i]=minus_v(first[i],first[0]); b[i]=minus_v(second[i],first[0]);
    }
    for (int i=0;i<3;++i) {ea[i]=minus_v(a[(i+1)%3],a[i]); eb[i]=minus_v(b[(i+1)%3],b[i]);}
    const V na=cross_v(ea[0],ea[1]),nb=cross_v(eb[0],eb[1]);
    if (separates(na,a,b) || separates(nb,a,b)) return false;
    for (int i=0;i<3;++i) {
        // In-plane axes also separate disjoint coplanar triangles.
        if (separates(cross_v(na,ea[i]),a,b) || separates(cross_v(nb,eb[i]),a,b)) return false;
        for (int j=0;j<3;++j) if (separates(cross_v(ea[i],eb[j]),a,b)) return false;
    }
    return true;
}
__device__ V transform(const double* m,V p) {
    return make_double3(m[0]*p.x+m[1]*p.y+m[2]*p.z+m[3],m[4]*p.x+m[5]*p.y+m[6]*p.z+m[7],
                        m[8]*p.x+m[9]*p.y+m[10]*p.z+m[11]);
}
}
extern "C" __global__ void leaf_candidates(const int* leaf_ids,int leaves,const double* source_bounds,
        const double* target_bounds,const int* target_links,int nodes,
        const double* pose,int2* candidates,int capacity,int* counts) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=leaves) return;
    const int leaf=leaf_ids[i]; const double* box=source_bounds+leaf*6;
    double lo[3],hi[3];
    for(int k=0;k<3;++k) {
        const double* row=pose+4*k;
        double lower=row[3],upper=row[3],absolute=fabs(row[3]);
        for(int j=0;j<3;++j) {
            const double p=row[j]*box[j],q=row[j]*box[3+j];
            lower+=fmin(p,q);upper+=fmax(p,q);absolute+=fmax(fabs(p),fabs(q));
        }
        const double margin=16*2.2204460492503131e-16*absolute;
        lo[k]=lower-margin;hi[k]=upper+margin;
    }
    for(int b=0;b<nodes;) {
        const double* target=target_bounds+b*6;
        const int* link=target_links+b*3;
        if (hi[0]<target[0] || lo[0]>target[3] || hi[1]<target[1] || lo[1]>target[4] ||
            hi[2]<target[2] || lo[2]>target[5]) {b=link[0];continue;}
        if (link[2]) {
            int slot=atomicAdd(counts+1,1);
            if (slot<capacity) candidates[slot]=make_int2(leaf,b);
        }
        ++b;
    }
}
extern "C" __global__ void intersect_leaves(const V* source,const int* source_ids,const int* source_links,
        const V* target,const int* target_ids,const int* target_links,const double* pose,
        const int2* candidates,int candidate_capacity,int2* hits,int capacity,int* counts) {
    const int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    const int n=counts[1]<candidate_capacity?counts[1]:candidate_capacity;
    for(int slot=warp;slot<n;slot+=gridDim.x*blockDim.x/32) {
        const int2 leaf=candidates[slot];
        const int* a=source_links+3*leaf.x;const int* b=target_links+3*leaf.y;
        for(int k=lane;k<a[2]*b[2];k+=32) {
            const int i=a[1]+k/b[2],j=b[1]+k%b[2];
            V tri[3];
            for(int v=0;v<3;++v) tri[v]=transform(pose,source[3*i+v]);
            const V* other=target+3*j;
            if (max3(tri[0].x,tri[1].x,tri[2].x)<min3(other[0].x,other[1].x,other[2].x) ||
                max3(other[0].x,other[1].x,other[2].x)<min3(tri[0].x,tri[1].x,tri[2].x) ||
                max3(tri[0].y,tri[1].y,tri[2].y)<min3(other[0].y,other[1].y,other[2].y) ||
                max3(other[0].y,other[1].y,other[2].y)<min3(tri[0].y,tri[1].y,tri[2].y) ||
                max3(tri[0].z,tri[1].z,tri[2].z)<min3(other[0].z,other[1].z,other[2].z) ||
                max3(other[0].z,other[1].z,other[2].z)<min3(tri[0].z,tri[1].z,tri[2].z)) continue;
            if (!intersect(tri,other)) continue;
            int index=atomicAdd(counts,1);
            if(index<capacity) hits[index]=make_int2(source_ids[i],target_ids[j]);
        }
    }
}
