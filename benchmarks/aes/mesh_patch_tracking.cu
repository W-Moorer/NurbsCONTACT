// Appended to mesh_pair.cu. A box stores center, half widths and three unit axes.
struct PatchBox { double3 center, half, axis[3]; };

__device__ double3 apply_rotation(const double* r,double3 p) {
    return make_double3(r[0]*p.x+r[1]*p.y+r[2]*p.z,
                        r[4]*p.x+r[5]*p.y+r[6]*p.z,
                        r[8]*p.x+r[9]*p.y+r[10]*p.z);
}
__device__ double3 place(const double* r,double3 p) {
    double3 q=apply_rotation(r,p);
    return make_double3(q.x+r[3],q.y+r[7],q.z+r[11]);
}
__device__ double support_radius(const PatchBox& box,double3 axis) {
    return fabs(prod(box.axis[0],axis))*box.half.x
          +fabs(prod(box.axis[1],axis))*box.half.y+fabs(prod(box.axis[2],axis))*box.half.z;
}
__device__ bool boxes_apart(const PatchBox& a,const PatchBox& b,double margin) {
    double3 d=sub3(b.center,a.center);
    double ha[3]={a.half.x,a.half.y,a.half.z}, hb[3]={b.half.x,b.half.y,b.half.z};
    for(int k=0;k<3;++k) {
        if(fabs(prod(d,a.axis[k])) > ha[k]+support_radius(b,a.axis[k])+margin ||
           fabs(prod(d,b.axis[k])) > hb[k]+support_radius(a,b.axis[k])+margin) return true;
    }
    return false;
}

// Only the current relative transform and the previous search origin are used.
extern "C" __global__ void refresh_gate(const double* pose,double* anchor,
    const double3* corners,double guard,unsigned long long* state) {
    bool rebuild=guard<=0 || state[2]>1;
    for(int i=0;i<8 && !rebuild;++i) {
        double3 d=sub3(place(pose,corners[i]),place(anchor,corners[i]));
        rebuild=prod(d,d)>guard*guard;
    }
    state[1]=0; state[2]=int(rebuild);
    if(rebuild) {
        state[0]=0;
        for(int i=0;i<12;++i) anchor[i]=pose[i];
    }
}
extern "C" __global__ void place_patches(const PatchBox* local,PatchBox* moved,int n,const double* pose) {
    int j=blockDim.x*blockIdx.x+threadIdx.x;
    if(j>=n) return;
    PatchBox b=local[j];
    b.center=place(pose,b.center);
    for(int k=0;k<3;++k) b.axis[k]=apply_rotation(pose,b.axis[k]);
    moved[j]=b;
}

extern "C" __global__ void check_envelopes(const PatchBox* current,const PatchBox* envelope,
                                           int n,unsigned long long* state,double pad) {
    int j=blockDim.x*blockIdx.x+threadIdx.x;
    if(j>=n) return;
    const PatchBox e=envelope[j], b=current[j];
    const double widths[3]={e.half.x,e.half.y,e.half.z};
    const double3 d=sub3(b.center,e.center);
    for(int k=0;k<3;++k) {
        if(fabs(prod(d,e.axis[k]))+support_radius(b,e.axis[k])+pad>widths[k]) {
            atomicExch(state+2,1ULL); return;
        }
    }
}

// Extrapolation chooses the shape of a search region. Containment is verified
// at every subsequent discrete pose before that region may be reused.
__device__ double3 rotation_increment(double3 v,const double* previous,const double* pose) {
    double3 local=make_double3(previous[0]*v.x+previous[4]*v.y+previous[8]*v.z,
                               previous[1]*v.x+previous[5]*v.y+previous[9]*v.z,
                               previous[2]*v.x+previous[6]*v.y+previous[10]*v.z);
    return apply_rotation(pose,local);
}
extern "C" __global__ void prepare_envelopes(const PatchBox* current,
    PatchBox* envelope,int n,const double* previous,const double* pose,int steps,double slack,
    const unsigned long long* state) {
    int j=blockDim.x*blockIdx.x+threadIdx.x;
    if(j>=n || !state[2]) return;
    const PatchBox b=current[j];
    PatchBox predicted=b, result=b;
    double3 center=make_double3(pose[3],pose[7],pose[11]);
    double3 travel=sub3(center,make_double3(previous[3],previous[7],previous[11]));
    double3 relative=sub3(b.center,center);
    double lo[3]={-b.half.x,-b.half.y,-b.half.z}, hi[3]={b.half.x,b.half.y,b.half.z}, width[3];
    for(int t=1;t<=steps;++t) {
        relative=rotation_increment(relative,previous,pose);
        for(int k=0;k<3;++k) predicted.axis[k]=rotation_increment(predicted.axis[k],previous,pose);
        double3 dc=step3(sub3(relative,sub3(b.center,center)),t,travel);
        for(int k=0;k<3;++k) {
            double shift=prod(dc,b.axis[k]), radius=support_radius(predicted,b.axis[k]);
            lo[k]=fmin(lo[k],shift-radius); hi[k]=fmax(hi[k],shift+radius);
        }
    }
    for(int k=0;k<3;++k) {
        result.center=step3(result.center,(lo[k]+hi[k])*.5,b.axis[k]);
        width[k]=(hi[k]-lo[k])*.5+slack;
    }
    result.half=make_double3(width[0],width[1],width[2]);
    envelope[j]=result;
}
extern "C" __global__ void begin_envelope_check(unsigned long long* state,int first) {
    state[1]=0; state[2]=first;
}
extern "C" __global__ void begin_discovery(unsigned long long* state) {
    if(state[2]) state[0]=0;
}
extern "C" __global__ void place_mesh(const double3* tri,double3* moved,double* box,int n,const double* pose) {
    int j=blockDim.x*blockIdx.x+threadIdx.x;
    if(j>=n) return;
    double lo[3]={PAIR_INFINITY,PAIR_INFINITY,PAIR_INFINITY}, hi[3]={-PAIR_INFINITY,-PAIR_INFINITY,-PAIR_INFINITY};
    for(int i=0;i<3;++i) {
        double3 p=place(pose,tri[j*3+i]); moved[j*3+i]=p;
        double x[3]={p.x,p.y,p.z};
        for(int k=0;k<3;++k) {lo[k]=fmin(lo[k],x[k]); hi[k]=fmax(hi[k],x[k]);}
    }
    for(int k=0;k<3;++k){box[j*6+k]=lo[k]; box[j*6+k+3]=hi[k];}
}
extern "C" __global__ void discover_patches(const int2* tree,const double* bounds,
    const PatchBox* a,const PatchBox* moved,int n,double margin,unsigned long long* state,int2* pairs,int capacity) {
    int j=blockDim.x*blockIdx.x+threadIdx.x;
    if(!state[2] || j>=n) return;
    PatchBox b=moved[j];
    double center[3]={b.center.x,b.center.y,b.center.z}, lo[3],hi[3];
    double3 axes[3]={make_double3(1,0,0),make_double3(0,1,0),make_double3(0,0,1)};
    for(int k=0;k<3;++k) {
        double radius=support_radius(b,axes[k])+margin;
        lo[k]=center[k]-radius; hi[k]=center[k]+radius;
    }
    for(int i=0;i<tree[0].x;) {
        int2 node=tree[i]; const double* v=bounds+6*i;
        if(v[0]>hi[0] || v[1]>hi[1] || v[2]>hi[2] || v[3]<lo[0] || v[4]<lo[1] || v[5]<lo[2]) {
            i=node.x; continue;
        }
        ++i;
        if(node.y<0 || boxes_apart(a[node.y],b,margin)) continue;
        unsigned long long k=atomicAdd(state,1ULL);
        if(k<capacity) pairs[k]=make_int2(node.y,j);
    }
}

extern "C" __global__ void intersect_patches(const int2* pairs,int capacity,unsigned long long* state,
    const PatchBox* obb_a,const PatchBox* obb_b,const int* oa,const int* ob,
    const double3* a,const double3* b,const double* box_a,const double* box_b,
    const int* ia,const int* ib,double padding,double touch_sq,int2* hits,unsigned long long hit_capacity) {
    if(state[0]>(unsigned long long)capacity) return;
    int n=(int)state[0];
    for(int row=blockIdx.x;row<n;row+=gridDim.x) {
        int2 pair=pairs[row];
        if(boxes_apart(obb_a[pair.x],obb_b[pair.y],padding)) continue;
        int sa=oa[pair.x], na=oa[pair.x+1]-sa, sb=ob[pair.y], nb=ob[pair.y+1]-sb;
        for(int k=threadIdx.x;k<na*nb;k+=blockDim.x) {
            int i=sa+k%na, j=sb+k/na;
            const double* bb=box_b+6*j;
            if(boxes_sq(box_a+6*i,bb,bb+3)>padding*padding) continue;
            if(triangles_overlap(a+3*i,b+3*j,touch_sq)) {
                unsigned long long id=atomicAdd(state+1,1ULL);
                if(id<hit_capacity) hits[id]=make_int2(ia[i],ib[j]);
            }
        }
    }
}

extern "C" __global__ void mark_faces(const int2* hits,int n,const int* ia,const int* ib,
                                      int columns,unsigned* marked) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n) return;
    int2 hit=hits[i];
    int key=ia[hit.x]*columns+ib[hit.y];
    atomicOr(marked+key/32,1u<<(key%32));
}
__device__ void emit_face_keys(unsigned bits,int first,int columns,const unsigned* tags_a,
                               const unsigned* tags_b,int* count,unsigned long long* out) {
    if(!bits) return;
    int slot=atomicAdd(count,__popc(bits));
    while(bits) {
        int key=first+__ffs(bits)-1;
        out[slot++]=((unsigned long long)tags_a[key/columns]<<32)|tags_b[key%columns];
        bits&=bits-1;
    }
}
extern "C" __global__ void face_changes(unsigned* marked,unsigned* previous,int words,
    int columns,const unsigned* tags_a,const unsigned* tags_b,int* counts,
    unsigned long long* current,unsigned long long* entered,unsigned long long* left) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=words) return;
    unsigned now=marked[i], old=previous[i];
    previous[i]=now;
    marked[i]=0;
    emit_face_keys(now,32*i,columns,tags_a,tags_b,counts,current);
    emit_face_keys(now&~old,32*i,columns,tags_a,tags_b,counts+1,entered);
    emit_face_keys(old&~now,32*i,columns,tags_a,tags_b,counts+2,left);
}
