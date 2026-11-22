// Coordinates and distances in millimeters. The host prepends TRIANGLE_CUDA.
#define PAIR_INFINITY (__longlong_as_double(0x7ff0000000000000LL))
__device__ double3 cross_pair(double3 a, double3 b) {
    return make_double3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}

__device__ bool axis_separates(const double3* a, const double3* b, double3 axis, double tol_sq) {
    double p = prod(axis, sub3(a[1], a[0])), q = prod(axis, sub3(a[2], a[0]));
    double amin = fmin(0., fmin(p,q)), amax = fmax(0., fmax(p,q));
    p = prod(axis, sub3(b[0], a[0]));
    q = prod(axis, sub3(b[1], a[0]));
    double r = prod(axis, sub3(b[2], a[0]));
    double gap = fmax(amin-fmax(p,fmax(q,r)), fmin(p,fmin(q,r))-amax);
    return gap>0 && gap*gap>tol_sq*prod(axis,axis);
}

__device__ bool triangles_overlap(const double3* a, const double3* b, double tol_sq) {
    double3 ea[3], eb[3];
    for (int i=0; i<3; ++i) {
        ea[i] = sub3(a[(i+1)%3], a[i]);
        eb[i] = sub3(b[(i+1)%3], b[i]);
    }
    double3 na = cross_pair(ea[0], ea[1]), nb = cross_pair(eb[0], eb[1]);
    if (axis_separates(a,b,na,tol_sq) || axis_separates(a,b,nb,tol_sq)) return false;
    for (int i=0; i<3; ++i) {
        // The in-plane axes also distinguish disjoint coplanar triangles.
        if (axis_separates(a,b,cross_pair(na,ea[i]),tol_sq) ||
            axis_separates(a,b,cross_pair(nb,eb[i]),tol_sq)) return false;
        for (int j=0; j<3; ++j) {
            if (axis_separates(a,b,cross_pair(ea[i],eb[j]),tol_sq)) return false;
        }
    }
    return true;
}

__device__ bool crosses_face(double3 a, double3 b, const double3* tri) {
    double3 u = sub3(tri[1], tri[0]), v = sub3(tri[2], tri[0]);
    double3 n = cross_pair(u, v), direction = sub3(b, a);
    double denom = prod(n, direction);
    if (denom == 0) return false;
    double s = prod(n, sub3(tri[0], a)) / denom;
    if (s < 0 || s > 1) return false;
    double3 w = sub3(step3(a, s, direction), tri[0]);
    double x = prod(cross_pair(w, v), n), y = prod(cross_pair(u, w), n);
    return x >= 0 && y >= 0 && x+y <= prod(n, n);
}

__device__ double triangle_pair_sq(const double3* a, const double3* b) {
    double best = PAIR_INFINITY;
    for (int i=0; i<3; ++i) {
        int next = (i+1)%3;
        if (crosses_face(a[i], a[next], b) || crosses_face(b[i], b[next], a)) return 0;
        best = fmin(best, fmin(distance2(a[i], b), distance2(b[i], a)));
        double3 u = sub3(a[next], a[i]);
        for (int j=0; j<3; ++j) {
            double3 v = sub3(b[(j+1)%3], b[j]), w = sub3(a[i], b[j]);
            double3 n = cross_pair(u, v);
            double denom = prod(n, n);
            if (denom == 0) continue;
            double s = prod(cross_pair(v, w), n) / denom;
            double t = prod(cross_pair(u, w), n) / denom;
            if (s>=0 && s<=1 && t>=0 && t<=1) {
                double3 d = sub3(step3(a[i], s, u), step3(b[j], t, v));
                best = fmin(best, prod(d, d));
            }
        }
    }
    return best;
}

__device__ double boxes_sq(const double* box, const double* lo, const double* hi) {
    double d2 = 0;
    for (int k=0; k<3; ++k) {
        double gap = fmax(0., fmax(lo[k]-box[3+k], box[k]-hi[k]));
        d2 += gap*gap;
    }
    return d2;
}

extern "C" __global__ void near_pairs(const int4* nodes, const double* boxes,
    const double* tri_boxes, const double3* a, const double3* local_b,
    int start, int stop, const double* pose, double cutoff_sq, double box_limit_sq,
    double touch_sq, int overlaps_only, double* nearest_sq, int* near_count, int* touch_count,
    int2* ids, int* id_count, int capacity) {
    int id = start + blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= stop) return;
    double3 b[3];
    double lo[3] = {PAIR_INFINITY, PAIR_INFINITY, PAIR_INFINITY};
    double hi[3] = {-PAIR_INFINITY, -PAIR_INFINITY, -PAIR_INFINITY};
    for (int i=0; i<3; ++i) {
        const double3 p = local_b[3*id+i];
        double xyz[3];
        for (int k=0; k<3; ++k) {
            const double* r = pose + 4*k;
            xyz[k] = r[0]*p.x+r[1]*p.y+r[2]*p.z+r[3];
            lo[k] = fmin(lo[k], xyz[k]);
            hi[k] = fmax(hi[k], xyz[k]);
        }
        b[i] = make_double3(xyz[0], xyz[1], xyz[2]);
    }
    double best = PAIR_INFINITY;
    int hits = 0, touches = 0;
    for (int i=0; i<nodes[0].x;) {
        int4 node = nodes[i];
        if (boxes_sq(boxes+6*i, lo, hi)>box_limit_sq) { i=node.x; continue; }
        ++i;
        for (int j=node.y; j>=0 && j<node.z; ++j) {
            if (boxes_sq(tri_boxes+6*j, lo, hi)>box_limit_sq) continue;
            if (overlaps_only) {
                if (triangles_overlap(a+3*j, b, touch_sq)) {
                    ++touches;
                    if (ids) {
                        int slot=atomicAdd(id_count,1);
                        if (slot<capacity) ids[slot]=make_int2(j,id);
                    }
                }
                continue;
            }
            double sq = triangle_pair_sq(a+3*j, b);
            if (sq <= cutoff_sq) {
                ++hits;
                touches += sq <= touch_sq;
                best = fmin(best, sq);
            }
        }
    }
    nearest_sq[id] = best;
    near_count[id] = hits;
    touch_count[id] = touches;
}
