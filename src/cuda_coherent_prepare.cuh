// Immutable evaluation and enclosure caches. Setup is excluded from online time.
__global__ void prepare_rule_envelopes(BodyView body, RuleEnvelope *out, int count) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const int face = k / rule_envelopes;
    int block = k % rule_envelopes, order = 12;
    while (block >= (order/4)*(order/4)) { block -= (order/4)*(order/4); order *= 2; }
    RuleEnvelope e{};
    for (int d=0;d<3;++d) { e.points.lo.x[d]=e.normals.lo.x[d]=HUGE_VAL; e.points.hi.x[d]=e.normals.hi.x[d]=-HUGE_VAL; }
    const int u=4*(block/(order/4)), v=4*(block%(order/4));
    for (int i=0;i<4;++i) for (int j=0;j<4;++j) {
        int ij=(u+i)*order+v+j;
        WeightedSite s{};
        if(order<=24 && body.g.charts[face].rule_start>=0)
            s=body.g.rules[body.g.charts[face].rule_start+(order==12?0:144)+ij];
        else if(order>=48)
            s=body.refined_rules[face*refined_rule_sites+(order==48?0:48*48)+ij];
        else {
            const auto &f=body.g.charts[face]; const auto r=f.domain;
            const double du=(r.U-r.u)/order, dv=(r.Vv-r.v)/order;
            const double U=r.u+(ij/order+.5)*du,V=r.v+(ij%order+.5)*dv;
            Jet jet;
            if(face_contains(body.g,f,U,V)>0 && face_jet(body.g,face,U,V,jet)) {
                const auto n=cross(jet.u,jet.v); const double a=norm(n);
                if(a>0) s={jet.p,scale(n,f.orientation/a),a*du*dv};
            }
        }
        if (!(s.area>0)) continue;
        e.valid=1;
        for(int d=0;d<3;++d) {
            e.points.lo.x[d]=fmin(e.points.lo.x[d],s.point.x[d]); e.points.hi.x[d]=fmax(e.points.hi.x[d],s.point.x[d]);
            e.normals.lo.x[d]=fmin(e.normals.lo.x[d],s.normal.x[d]); e.normals.hi.x[d]=fmax(e.normals.hi.x[d],s.normal.x[d]);
        }
    }
    out[k]=e;
}
// Conservative geometric neighborhoods, not a claim of topological adjacency.
// They only prioritize proofs; no remote edge is removed from the search.
__global__ void prepare_chart_neighbors(View g, unsigned char *out, int count) {
    const size_t k=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(k>=static_cast<size_t>(count)*count) return;
    const auto &a=g.charts[k/count], &b=g.charts[k%count];
    out[k]=overlaps(a.box,b.box,4*(a.tol+b.tol)+1e-6);
}

// Merge each deterministic search group in small ordered batches. Duplicate
// sources are marked once before the nonlinear work, without changing priority.
