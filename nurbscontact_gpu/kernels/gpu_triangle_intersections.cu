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
__device__ void collect_leaves(const int *leaf_ids, int leaves,
                               const double *source_bounds,
                               const float4 *target_bounds, int nodes,
                               const double *pose, int2 *candidates,
                               int capacity, int *counts, double skin = 0) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= leaves)
    return;
  const int leaf = leaf_ids[i];
  const double *box = source_bounds + leaf * 6;
  float lo[3], hi[3];
  for (int k = 0; k < 3; ++k) {
    const double *row = pose + 4 * k;
    double lower = row[3], upper = row[3], absolute = fabs(row[3]);
    for (int j = 0; j < 3; ++j) {
      const double p = row[j] * box[j], q = row[j] * box[3 + j];
      lower += fmin(p, q);
      upper += fmax(p, q);
      absolute += fmax(fabs(p), fabs(q));
    }
    const double margin =
        skin + 16 * 2.2204460492503131e-16 * (absolute + skin);
    lo[k] = __double2float_rd(lower - margin);
    hi[k] = __double2float_ru(upper + margin);
  }
  const float4 root_lo = target_bounds[0], root_hi = target_bounds[1];
  if (hi[0] < root_lo.x || lo[0] > root_hi.x || hi[1] < root_lo.y ||
      lo[1] > root_hi.y || hi[2] < root_lo.z || lo[2] > root_hi.z)
    return;
  for (int b = 0; b < nodes;) {
    const float4 low = target_bounds[2 * b], high = target_bounds[2 * b + 1];
    if (hi[0] < low.x || lo[0] > high.x || hi[1] < low.y || lo[1] > high.y ||
        hi[2] < low.z || lo[2] > high.z) {
      b = __float_as_int(low.w);
      continue;
    }
    if (__float_as_int(high.w)) {
      int slot = atomicAdd(counts + 1, 1);
      if (slot < capacity)
        candidates[slot] = make_int2(leaf, b);
    }
    ++b;
  }
}
extern "C" __global__ void leaf_candidates(const int *leaves, int n,
                                           const double *a, const float4 *b,
                                           int nodes, const double *pose,
                                           int2 *candidates, int capacity,
                                           int *totals) {
  collect_leaves(leaves, n, a, b, nodes, pose, candidates, capacity, totals);
}
namespace {
__device__ float3 fdiff(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 fcross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}
__device__ float fdot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ bool fgaps(float a, float b, float c, float d, float e, float f,
                      float pad) {
  return fminf(a, fminf(b, c)) - fmaxf(d, fmaxf(e, f)) > pad ||
         fminf(d, fminf(e, f)) - fmaxf(a, fmaxf(b, c)) > pad;
}
// Any direction can prove disjointness. Enclose all six projections so a
// rounded normal never has to be an exact separating-axis direction.
__device__ bool fplane(float3 n, const float3 a[3], const float3 b[3],
                       float error) {
  float scale = fabsf(n.x) + fabsf(n.y) + fabsf(n.z);
  if (!(scale > 1e-18f && scale < 1e18f))
    return false;
  return fgaps(fdot(n, a[0]), fdot(n, a[1]), fdot(n, a[2]), fdot(n, b[0]),
               fdot(n, b[1]), fdot(n, b[2]), error * scale + 1e-30f);
}
__device__ float2 project2(float3 p, int drop) {
  return drop == 0
             ? make_float2(p.y, p.z)
             : (drop == 1 ? make_float2(p.z, p.x) : make_float2(p.x, p.y));
}
__device__ bool empty_projection(const float3 a[3], const float3 b[3],
                                 float3 normal, float pad) {
  const float x = fabsf(normal.x), y = fabsf(normal.y), z = fabsf(normal.z);
  const int drop = x > y && x > z ? 0 : (y > z ? 1 : 2);
  float2 p[3], q[3];
  for (int i = 0; i < 3; ++i) {
    p[i] = project2(a[i], drop);
    q[i] = project2(b[i], drop);
  }
  for (int i = 0; i < 6; ++i) {
    const float2 *f = i < 3 ? p : q;
    const float2 start = f[i % 3], end = f[(i + 1) % 3];
    const float nx = start.y - end.y, ny = end.x - start.x;
    const float bound = pad * (fabsf(nx) + fabsf(ny)) + 1e-30f;
    if (fgaps(nx * p[0].x + ny * p[0].y, nx * p[1].x + ny * p[1].y,
              nx * p[2].x + ny * p[2].y, nx * q[0].x + ny * q[0].y,
              nx * q[1].x + ny * q[1].y, nx * q[2].x + ny * q[2].y, bound))
      return true;
  }
  return false;
}
__device__ float3 move_proxy(float3 p, const double *m) {
  return make_float3(
      float(m[0]) * p.x + float(m[1]) * p.y + float(m[2]) * p.z + float(m[3]),
      float(m[4]) * p.x + float(m[5]) * p.y + float(m[6]) * p.z + float(m[7]),
      float(m[8]) * p.x + float(m[9]) * p.y + float(m[10]) * p.z +
          float(m[11]));
}
__device__ bool separated_proxies(const float3 *a, const float3 *b,
                                  float error) {
  // Very large coordinates could overflow products in the single-precision
  // projections. Leave their classification to the double-precision path.
  if (!(error < 1e5f))
    return false;
  if (fgaps(a[0].x, a[1].x, a[2].x, b[0].x, b[1].x, b[2].x, error) ||
      fgaps(a[0].y, a[1].y, a[2].y, b[0].y, b[1].y, b[2].y, error) ||
      fgaps(a[0].z, a[1].z, a[2].z, b[0].z, b[1].z, b[2].z, error))
    return true;
  const float3 na = fcross(fdiff(a[1], a[0]), fdiff(a[2], a[0]));
  return fplane(na, a, b, error) ||
         fplane(fcross(fdiff(b[1], b[0]), fdiff(b[2], b[0])), a, b, error) ||
         empty_projection(a, b, na, error);
}
__device__ bool proxy_excludes(const float3 *a, const float3 *b,
                               const double *m, float error) {
  const float3 placed[3] = {move_proxy(a[0], m), move_proxy(a[1], m),
                            move_proxy(a[2], m)};
  return separated_proxies(placed, b, error);
}
} // namespace
__device__ bool finite_hit(const V tri[3], const V *other) {
  if (max3(tri[0].x, tri[1].x, tri[2].x) <
          min3(other[0].x, other[1].x, other[2].x) ||
      max3(other[0].x, other[1].x, other[2].x) <
          min3(tri[0].x, tri[1].x, tri[2].x) ||
      max3(tri[0].y, tri[1].y, tri[2].y) <
          min3(other[0].y, other[1].y, other[2].y) ||
      max3(other[0].y, other[1].y, other[2].y) <
          min3(tri[0].y, tri[1].y, tri[2].y) ||
      max3(tri[0].z, tri[1].z, tri[2].z) <
          min3(other[0].z, other[1].z, other[2].z) ||
      max3(other[0].z, other[1].z, other[2].z) <
          min3(tri[0].z, tri[1].z, tri[2].z))
    return false;
  return intersect(tri, other);
}
__device__ void test_leaves(const V *source, const int *source_ids,
                            const int *source_links, const V *target,
                            const int *target_ids, const int *target_links,
                            const double *pose, const float3 *source_proxy,
                            const float3 *target_proxy, double extent,
                            int width, const int2 *candidates,
                            int candidate_capacity, int2 *hits, int capacity,
                            int *counts) {
  const int worker = (blockIdx.x * blockDim.x + threadIdx.x) / width,
            lane = threadIdx.x % width;
  const int n = counts[1] < candidate_capacity ? counts[1] : candidate_capacity;
  // A rigid transform has unit-length rows. Include conversion, placement,
  // scalar products and the final float comparison in each projection pad.
  const float error = __double2float_ru(
      128 * 1.1920928955078125e-7 *
          (4 * extent + fabs(pose[3]) + fabs(pose[7]) + fabs(pose[11])) +
      1e-30);
  for (int slot = worker; slot < n; slot += gridDim.x * blockDim.x / width) {
    const int2 leaf = candidates[slot];
    const int *a = source_links + 3 * leaf.x;
    const int *b = target_links + 3 * leaf.y;
    for (int k = lane; k < a[2] * b[2]; k += width) {
      const int i = a[1] + k / b[2], j = b[1] + k % b[2];
      if (proxy_excludes(source_proxy + 3 * i, target_proxy + 3 * j, pose,
                         error))
        continue;
      V tri[3];
      for (int v = 0; v < 3; ++v)
        tri[v] = transform(pose, source[3 * i + v]);
      const V *other = target + 3 * j;
      if (!finite_hit(tri, other))
        continue;
      int index = atomicAdd(counts, 1);
      if (index < capacity)
        hits[index] = make_int2(source_ids[i], target_ids[j]);
    }
  }
}

extern "C" __global__ void intersect_leaves(
    const V *source, const int *source_ids, const int *source_links,
    const V *target, const int *target_ids, const int *target_links,
    const double *pose, const float3 *source_proxy, const float3 *target_proxy,
    double extent, int width, const int2 *candidates, int candidate_capacity,
    int2 *hits, int capacity, int *counts) {
  test_leaves(source, source_ids, source_links, target, target_ids,
              target_links, pose, source_proxy, target_proxy, extent, width,
              candidates, candidate_capacity, hits, capacity, counts);
}

struct MeshPairView {
  const int *leaves;
  const double *bounds;
  const float4 *traversal;
  const V *a;
  const int *a_ids;
  const int *a_links;
  const V *b;
  const int *b_ids;
  const int *b_links;
  const float3 *a_proxy;
  const float3 *b_proxy;
  int2 *candidates;
  int2 *hits;
  int2 *filtered;
  double extent;
  int leaves_n, nodes, candidate_n, hit_n, width, filtered_n;
};
// Each cached list encloses all placements within 'skin' coordinate-wise
// displacement of the source solid. Only poses supplied by the caller are used.
extern "C" __global__ void check_candidate_cache(const MeshPairView *pairs,
                                                 const double *poses,
                                                 int *counts, int *filtered,
                                                 int *refresh, double *anchors,
                                                 double skin) {
  const int pair = blockIdx.y;
  if (threadIdx.x)
    return;
  const double *m = poses + 12 * pair;
  double *old = anchors + 12 * pair;
  const double *box = pairs[pair].bounds;
  bool rebuild = !isfinite(old[0]);
  for (int k = 0; k < 3 && !rebuild; ++k) {
    double motion = fabs(m[4 * k + 3] - old[4 * k + 3]);
    for (int j = 0; j < 3; ++j)
      motion += fabs(m[4 * k + j] - old[4 * k + j]) *
                fmax(fabs(box[j]), fabs(box[3 + j]));
    rebuild = motion > skin * (1 - 16 * 2.2204460492503131e-16);
  }
  refresh[pair] = rebuild;
  counts[2 * pair] = 0;
  if (rebuild) {
    for (int j = 0; j < 12; ++j)
      old[j] = m[j];
    counts[2 * pair + 1] = 0;
    filtered[pair] = 0;
  }
}
extern "C" __global__ void
batch_leaf_candidates(const MeshPairView *pairs, const double *poses,
                      int *counts, const int *refresh, double skin) {
  const int i = blockIdx.y;
  if (skin > 0 && !refresh[i])
    return;
  const MeshPairView &p = pairs[i];
  collect_leaves(p.leaves, p.leaves_n, p.bounds, p.traversal, p.nodes,
                 poses + 12 * i, p.candidates, p.candidate_n, counts + 2 * i,
                 skin);
}
extern "C" __global__ void batch_filter(const MeshPairView *pairs,
                                        const double *poses, int *counts,
                                        int *sizes, const int *refresh,
                                        double skin) {
  const int pair = blockIdx.y;
  if (skin > 0 && !refresh[pair])
    return;
  const MeshPairView &p = pairs[pair];
  const double *m = poses + 12 * pair;
  const float pad = __double2float_ru(
      128 * 1.1920928955078125e-7 *
          (4 * p.extent + fabs(m[3]) + fabs(m[7]) + fabs(m[11])) +
      skin * (1 + 1e-5) + 1e-30);
  const int width = p.width;
  const int id = blockIdx.x * blockDim.x + threadIdx.x,
            lane = threadIdx.x % width;
  const int n = min(counts[2 * pair + 1], p.candidate_n);
  for (int slot = id / width; slot < n;
       slot += gridDim.x * blockDim.x / width) {
    const int2 leaf = p.candidates[slot];
    const int *a = p.a_links + 3 * leaf.x;
    const int *b = p.b_links + 3 * leaf.y;
    for (int k = lane; k < a[2] * b[2]; k += width) {
      const int i = a[1] + k / b[2], j = b[1] + k % b[2];
      if (proxy_excludes(p.a_proxy + 3 * i, p.b_proxy + 3 * j, m, pad))
        continue;
      const int pos = atomicAdd(sizes + pair, 1);
      if (pos < p.filtered_n)
        p.filtered[pos] = make_int2(i, j);
    }
  }
}
extern "C" __global__ void batch_intersections(const MeshPairView *pairs,
                                               const double *poses, int *counts,
                                               const int *sizes, double skin) {
  const int i = blockIdx.y;
  const MeshPairView &p = pairs[i];
  if (sizes[i] > p.filtered_n) {
    test_leaves(p.a, p.a_ids, p.a_links, p.b, p.b_ids, p.b_links,
                poses + 12 * i, p.a_proxy, p.b_proxy, p.extent, p.width,
                p.candidates, p.candidate_n, p.hits, p.hit_n, counts + 2 * i);
    return;
  }
  for (int slot = blockIdx.x * blockDim.x + threadIdx.x; slot < sizes[i];
       slot += gridDim.x * blockDim.x) {
    const int2 faces = p.filtered[slot];
    if (skin > 0) {
      const double *m = poses + 12 * i;
      const float error = __double2float_ru(
          128 * 1.1920928955078125e-7 *
              (4 * p.extent + fabs(m[3]) + fabs(m[7]) + fabs(m[11])) +
          1e-30);
      if (proxy_excludes(p.a_proxy + 3 * faces.x, p.b_proxy + 3 * faces.y, m,
                         error))
        continue;
    }
    V tri[3];
    for (int v = 0; v < 3; ++v)
      tri[v] = transform(poses + 12 * i, p.a[3 * faces.x + v]);
    if (!finite_hit(tri, p.b + 3 * faces.y))
      continue;
    int pos = atomicAdd(counts + 2 * i, 1);
    if (pos < p.hit_n)
      p.hits[pos] = make_int2(p.a_ids[faces.x], p.b_ids[faces.y]);
  }
}

namespace {
__device__ V plus_v(V a, V b) {
  return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ V times_v(V a, double s) {
  return make_double3(a.x * s, a.y * s, a.z * s);
}
__device__ V lerp_v(V a, V b, double t) {
  return plus_v(a, times_v(minus_v(b, a), t));
}
__device__ double norm_v(V a) { return sqrt(proj(a, a)); }

// Return the finite edge/plane intersections. Keep the signs of sub-ulp
// offsets around planar end caps; snapping them can turn a point into an edge.
__device__ int plane_slice(const V tri[3], V pivot, V plane_normal, V *out) {
  double height[3];
  for (int i = 0; i < 3; ++i)
    height[i] = proj(minus_v(tri[i], pivot), plane_normal);
  if (height[0] == 0 && height[1] == 0 && height[2] == 0)
    return 3;
  int count = 0;
  for (int i = 0; i < 3; ++i) {
    const int j = (i + 1) % 3;
    const double u = height[i], v = height[j];
    if (u == 0)
      out[count++] = tri[i];
    if ((u < 0 && v > 0) || (u > 0 && v < 0))
      out[count++] = lerp_v(tri[i], tri[j], u / (u - v));
  }
  return count;
}
__device__ double component(V p,int k){return k==0?p.x:(k==1?p.y:p.z);}
__device__ int coplanar_polygon(const V a[3],const V b[3],V normal,double eps,V* result) {
    V polygon[10],next[10];
    int n=3;
    for(int i=0;i<3;++i) polygon[i]=a[i];
    for(int edge=0;edge<3 && n;++edge) {
        const V o=b[edge],axis=cross_v(normal,minus_v(b[(edge+1)%3],o));
        const double tol=eps*norm_v(axis);
        int count=0;
        for(int i=0;i<n;++i) {
            const V p=polygon[i],q=polygon[(i+1)%n];
            double u=proj(minus_v(p,o),axis),v=proj(minus_v(q,o),axis);
            if(fabs(u)<=tol)u=0;
            if(fabs(v)<=tol)v=0;
            if(u>=0)next[count++]=p;
            if((u<0 && v>0)||(u>0 && v<0))next[count++]=lerp_v(p,q,u/(u-v));
        }
        n=count;
        for(int i=0;i<n;++i)polygon[i]=next[i];
    }
    int count=0;
    for(int i=0;i<n;++i) {
        bool duplicate=false;
        for(int j=0;j<count;++j) if(norm_v(minus_v(result[j],polygon[i]))<=eps)duplicate=true;
        if(!duplicate){if(count==6)return -1;result[count++]=polygon[i];}
    }
    return count ? count : -1;
}
} // namespace
// Counts give point, segment, or polygon size. A negative count flags an
// ill-conditioned construction. Hit IDs and both winding normals are preserved.
extern "C" __global__ void intersection_geometry(const V* source,const int* source_lookup,
        const V* target,const int* target_lookup,const double* pose,const int2* pairs,const int* totals,
        int capacity,int* sizes,V* points,V* normals) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=totals[0] || i>=capacity)return;
    const int2 hit=pairs[i];V a[3];const V* b=target+3*target_lookup[hit.y];
    for(int j=0;j<3;++j)a[j]=transform(pose,source[3*source_lookup[hit.x]+j]);
    V na=cross_v(minus_v(a[1],a[0]),minus_v(a[2],a[0]));
    V nb=cross_v(minus_v(b[1],b[0]),minus_v(b[2],b[0]));
    normals[2*i]=times_v(na,1/norm_v(na));normals[2*i+1]=times_v(nb,1/norm_v(nb));
    const double eps=64*2.2204460492503131e-16*(norm_v(a[0])+norm_v(b[0])+
                           norm_v(minus_v(a[1],a[0]))+norm_v(minus_v(b[1],b[0])));
    V* out=points+6*i;
    V x[3],y[3];
    const int nx=plane_slice(a,b[0],nb,x),ny=plane_slice(b,a[0],na,y);
    if(nx==3 && ny==3) {sizes[i]=coplanar_polygon(a,b,normals[2*i+1],eps,out);return;}
    if(nx<1 || ny<1 || nx==3 || ny==3) {sizes[i]=-1;return;}
    if(nx==1 || ny==1){out[0]=nx==1?x[0]:y[0];sizes[i]=1;return;}
    const V dx=minus_v(x[1],x[0]),dy=minus_v(y[1],y[0]);
    const V axis=proj(dx,dx)>proj(dy,dy)?dx:dy;
    int k=fabs(axis.x)>fabs(axis.y)?0:1;
    if(fabs(axis.z)>fabs(component(axis,k)))k=2;
    int ix=component(x[0],k)<component(x[1],k)?0:1;
    int iy=component(y[0],k)<component(y[1],k)?0:1;
    out[0]=component(x[ix],k)>component(y[iy],k)?x[ix]:y[iy];
    out[1]=component(x[1-ix],k)<component(y[1-iy],k)?x[1-ix]:y[1-iy];
    if(component(out[0],k)>component(out[1],k)+eps){sizes[i]=-1;return;}
    sizes[i]=norm_v(minus_v(out[0],out[1]))>eps?2:1;
}
