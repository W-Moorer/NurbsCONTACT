constexpr int coordinate_limit = 16;
struct ShaftState {
    double time, q[coordinate_limit], v[coordinate_limit], a[coordinate_limit], force[coordinate_limit];
};
struct Trial {
    ShaftState state;
    double jac[coordinate_limit * coordinate_limit], residual[coordinate_limit], error;
    double damping[coordinate_limit * coordinate_limit];
};
struct StepControl {
    ShaftState state;
    Trial working, next;
    double delta[coordinate_limit], qpred[coordinate_limit], vpred[coordinate_limit];
    int dispatch, invalid, status, last_step, current_pair;
};
struct ScalarGrid {
    __device__ int thread_rank() const { return 0; }
    __device__ void sync() const {}
};
struct DeviceScene {
    const AxisCoordinate *coordinates;
    const JointBody *bodies;
    const ContactLaw *pairs;
    const LoadPoint *loads;
    const BodyView *views;
    int n, b, np;
    PairScratch *cache;
    Workspace *workspace;
    StepControl *control;
    double *observed, *frames;
    unsigned long long *finished;
};
__device__ MatPose body_pose(const JointBody &b, const double *q) {
    MatPose out{b.origin, b.axis, b.angle};
    if (b.hinge >= 0) {
        double a = q[b.hinge];
        MatPose orbit{b.pivot, b.axis, a};
        out.origin = plus(b.pivot, rotate(orbit, sub(b.origin, b.pivot)));
        out.angle += a;
    }
    if (b.translation >= 0)
        out.origin = plus(out.origin, scale(b.direction, q[b.translation]));
    return out;
}
__device__ V angular(const JointBody &b, int id) { return id == b.hinge ? b.axis : V{}; }
__device__ V speed_gradient(const JointBody &b, MatPose p, V at_point, int id) {
    V x = id == b.translation ? b.direction : V{};
    if (id == b.hinge) {
        V arm = rotate({{}, b.axis, p.angle - b.angle}, sub(b.origin, b.pivot));
        x = plus(x, plus(cross(b.axis, arm), cross(b.axis, sub(at_point, p.origin))));
    }
    return x;
}
__device__ double interpolated(const AxisCoordinate &c, const LoadPoint *p, double t) {
    if (c.first == c.end)
        return 0;
    int a = c.first + 1, b = c.end - 1;
    while (a < b) {
        int m = (a + b) / 2;
        if (t < p[m].t)
            b = m;
        else
            a = m + 1;
    }
    auto l = p[a - 1], r = p[a];
    double x = fmin(1., fmax(0., (t - l.t) / (r.t - l.t)));
    if ((l.value <= 0 && r.value >= 0) || (l.value >= 0 && r.value <= 0))
        return (1 - x) * l.value + x * r.value;
    return x == 1 ? r.value : l.value + x * (r.value - l.value);
}
__device__ Region transport(const Anchor &h, MatPose a, MatPose b, int &invalid) {
    V x = world(a, h.a), y = world(b, h.b), n = plus(rotate(a, h.na), rotate(b, h.nb));
    double len = norm(n);
    if (len < 1 || !isfinite(len)) {
        invalid = 1;
        return {};
    }
    n = scale(n, 1 / len);
    return {scale(plus(x, y), .5), n, scale(plus(rotate(a, h.ca), rotate(b, h.cb)), .5), dot(sub(x, y), n)};
}
// Compare independently reconstructed regions at one state. Both wrenches are
// expressed in world coordinates and moments are about each body's own origin.
struct AuditLoad { V force, torque_a, torque_b; double normal; };
__device__ AuditLoad region_load(const Region &r, const ContactLaw &law, const DeviceScene &s,
                                 const ShaftState &state, MatPose a, MatPose b) {
    V p = world(a,scale(r.p,.001)), n = rotate(a,r.n), arm = rotate(a,scale(r.arm,.001));
    double depth = r.depth * .001, rate = 0;
    for (int j=0;j<s.n;++j)
        rate += dot(n,sub(speed_gradient(s.bodies[law.b],b,p,j),speed_gradient(s.bodies[law.a],a,p,j))) * state.v[j];
    double f = law.k * pow(depth,law.exponent) - law.rate * pow(depth,.5*(law.exponent-1))*rate;
    if (f<=0) return {};
    V force = scale(n,f), couple = scale(arm,f);
    return {force,scale(plus(cross(sub(p,a.origin),force),couple),-1),
                  plus(cross(sub(p,b.origin),force),couple),f};
}
__device__ __noinline__ void audit_regions(const DeviceScene &s,const ShaftState &state,const ContactLaw &law,
                                           MatPose a,MatPose b,int pair,bool initial) {
    auto &w=*s.workspace;
    auto &entry=w.audit[(initial ? 0 : s.control->last_step+1)*s.np+pair];
    auto &cache=s.cache[pair];
    ++entry.queries;
    entry.regions+=w.regions;
    entry.count_changes+=w.regions!=cache.audited_regions;
    cache.audited_regions=w.regions;
    // Both lists use the lexicographic order of the geometrically confirmed face pairs.
    int i=0,j=0;
    while (i<w.found && j<cache.audited_faces) {
        Faces now=w.witnesses[i].faces, old=cache.audit_previous[j];
        if (equal(now,old)) {++i; ++j;}
        else if (now.a<old.a || (now.a==old.a && now.b<old.b)) {++i; ++entry.entered_faces;}
        else {++j; ++entry.exited_faces;}
    }
    entry.entered_faces+=w.found-i;
    entry.exited_faces+=cache.audited_faces-j;
    cache.audited_faces=w.found;
    for (i=0;i<w.found;++i) cache.audit_previous[i]=w.witnesses[i].faces;
    int loaded=0,reference_loaded=0;
    for (i=0;i<w.regions;++i) {
        loaded+=region_load(w.result[i],law,s,state,a,b).normal>0;
        reference_loaded+=region_load(w.reference[i],law,s,state,a,b).normal>0;
    }
    if (loaded!=reference_loaded) {atomicCAS(w.status,0,10); return;}
    double weight=law.shared && loaded ? 1./loaded : 1.;
    for (i=0;i<w.regions;++i) {
        auto x=region_load(w.result[i],law,s,state,a,b), y=region_load(w.reference[i],law,s,state,a,b);
        if ((x.normal>0)!=(y.normal>0)) {atomicCAS(w.status,0,10); return;}
        for (int k=0;k<3;++k) {
            double f=fabs(weight*(x.force.x[k]-y.force.x[k]));
            double ma=fabs(weight*(x.torque_a.x[k]-y.torque_a.x[k]));
            double mb=fabs(weight*(x.torque_b.x[k]-y.torque_b.x[k]));
            if (!isfinite(f)||!isfinite(ma)||!isfinite(mb)) {atomicCAS(w.status,0,10);return;}
            entry.force_error[k]=fmax(entry.force_error[k],f);
            entry.torque_a_error[k]=fmax(entry.torque_a_error[k],ma);
            entry.torque_b_error[k]=fmax(entry.torque_b_error[k],mb);
        }
    }
}
template <class Grid>
__device__ __noinline__ void forces(Grid grid, const DeviceScene &s, Trial &out, double dt, bool initial) {
    int lane = grid.thread_rank();
    auto &control = *s.control;
    auto &w = *s.workspace;
    auto &state = out.state;
    if (!lane) {
        ++w.evaluations;
        control.invalid = 0;
        for (int i = 0; i < s.n; ++i) {
            const auto &c = s.coordinates[i];
            state.force[i] = c.load - c.k * (state.q[i] - c.rest) - c.c * state.v[i] +
                             interpolated(c, s.loads, state.time);
            for (int j = 0; j < s.n; ++j) {
                out.jac[i * s.n + j] = i == j ? c.k : 0;
                out.damping[i * s.n + j] = i == j ? c.c : 0;
            }
        }
    }
    grid.sync();
    for (int pair = 0; pair < s.np; ++pair) {
        const auto law = s.pairs[pair];
        const auto &ba = s.bodies[law.a], &bb = s.bodies[law.b];
        auto a = body_pose(ba, state.q), b = body_pose(bb, state.q);
        Query q{{s.views[law.a], s.views[law.b]}, {between(a, b), between(b, a)}, 2000 * law.max_depth, &w};
        auto &cache = s.cache[pair];
        if (!lane) {
            control.current_pair = pair;
            w.regions = 0;
            if (cache.ready)
                for (int k = 0; k < cache.anchors; ++k) {
                    Region r = transport(cache.anchor[k], a, b, control.invalid);
                    if (r.depth > 0)
                        w.result[w.regions++] = r;
                }
            w.selected = (!w.regions || w.regions != cache.anchors || !cache.ready);
        }
        grid.sync();
        // Save the dispatch decision per thread before entering collective functions.
        bool fresh = w.selected;
        grid.sync();
        if (fresh) {
            if constexpr (std::is_same_v<Grid, ScalarGrid>) {
                control.status = 9;
                return;
            } else {
                detect(grid, q, cache);
                if (*w.status)
                    return;
                unsigned long long stamp = lane ? 0 : timer_ns();
                integrate_regions(grid, q);
                if (!lane)
                    w.integral_ns += timer_ns() - stamp;
                if (*w.status)
                    return;
                if (!lane) {
                    if (w.audit) audit_regions(s,state,law,a,b,pair,initial);
                    for (int k = 0; k < w.regions; ++k) {
                        auto &r = w.result[k];
                        r.p = world(a, scale(r.p, .001));
                        r.n = rotate(a, r.n);
                        r.arm = rotate(a, scale(r.arm, .001));
                        r.depth *= .001;
                        if (!cache.ready) {
                            V half = scale(r.n, .5 * r.depth);
                            cache.anchor[k] = {local(a, plus(r.p, half)), local(b, sub(r.p, half)),
                                               rotate(a, r.n, -1),        rotate(b, r.n, -1),
                                               rotate(a, r.arm, -1),      rotate(b, r.arm, -1)};
                        }
                    }
                    if (!cache.ready) {
                        cache.anchors = w.regions;
                        cache.ready = 1;
                    }
                }
            }
        }
        grid.sync();
        if (!lane) {
            double maxdepth = 0, sum = 0, force[coordinate_limit]{},
                   jac[coordinate_limit * coordinate_limit]{}, damping[coordinate_limit * coordinate_limit]{};
            int loaded = 0;
            for (int k = 0; k < w.regions; ++k) {
                auto r = w.result[k];
                maxdepth = fmax(maxdepth, r.depth);
                if (r.depth > law.max_depth) {
                    control.invalid = 1;
                    break;
                }
                double gradient[coordinate_limit], rate = 0;
                for (int i = 0; i < s.n; ++i) {
                    gradient[i] = dot(r.n, sub(speed_gradient(bb, b, r.p, i), speed_gradient(ba, a, r.p, i)));
                    rate += gradient[i] * state.v[i];
                }
                double elastic = law.k * pow(r.depth, law.exponent), power = .5 * (law.exponent - 1);
                double c = law.rate * pow(r.depth, power), f = elastic - c * rate;
                if (f <= 0)
                    continue;
                ++loaded;
                sum += f;
                double k0 = (law.exponent * elastic - power * c * rate) / r.depth;
                for (int i = 0; i < s.n; ++i) {
                    double arm = gradient[i] + dot(r.arm, sub(angular(bb, i), angular(ba, i)));
                    force[i] += arm * f;
                    for (int j = 0; j < s.n; ++j) {
                        double product = arm * gradient[j];
                        jac[i * s.n + j] += k0 * product;
                        damping[i * s.n + j] += c * product;
                    }
                }
            }
            double weight = law.shared && loaded ? 1. / loaded : 1.;
            for (int i = 0; i < s.n; ++i) {
                state.force[i] += weight * force[i];
                for (int j = 0; j < s.n; ++j) {
                    out.jac[i * s.n + j] += weight * jac[i * s.n + j];
                    out.damping[i * s.n + j] += weight * damping[i * s.n + j];
                }
            }
            s.observed[3 * pair] = w.regions ? -maxdepth : NAN;
            s.observed[3 * pair + 1] = sum * weight;
            s.observed[3 * pair + 2] = w.regions != 0;
        }
        grid.sync();
    }
    if (!lane && !initial) {
        for (int k = 0; k < s.n * s.n; ++k)
            out.jac[k] = dt * dt * out.jac[k] + 1.5 * dt * out.damping[k];
        for (int i = 0; i < s.n; ++i)
            out.jac[i * s.n + i] += 2 * s.coordinates[i].mass;
        out.error = 0;
        for (int i = 0; i < s.n; ++i) {
            out.residual[i] = s.coordinates[i].mass * (2 * state.a[i] - control.state.a[i]) - state.force[i];
            if (!isfinite(out.residual[i]))
                control.invalid = 1;
            out.error = fmax(out.error, fabs(out.residual[i]));
        }
        if (control.invalid)
            out.error = INFINITY;
    }
    grid.sync();
}
__device__ bool cholesky(Trial &trial, int n, double *rhs) {
    auto *m = trial.jac;
    for (int i = 0; i < n; ++i) {
        rhs[i] = -trial.residual[i];
        for (int j = 0; j <= i; ++j) {
            double x = m[i * n + j];
            for (int k = 0; k < j; ++k)
                x -= m[i * n + k] * m[j * n + k];
            if (i == j) {
                if (!(x > 0) || !isfinite(x))
                    return false;
                m[i * n + j] = sqrt(x);
            } else
                m[i * n + j] = x / m[j * n + j];
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j)
            rhs[i] -= m[i * n + j] * rhs[j];
        rhs[i] /= m[i * n + i];
    }
    for (int i = n; i--;) {
        for (int j = i + 1; j < n; ++j)
            rhs[i] -= m[j * n + i] * rhs[j];
        rhs[i] /= m[i * n + i];
    }
    return true;
}
__device__ bool anchors_cover(const DeviceScene &s, const double *q) {
    for (int i = 0; i < s.np; ++i) {
        const auto &cache = s.cache[i];
        if (!cache.ready || !cache.anchors)
            return false;
        const auto law = s.pairs[i];
        const auto a = body_pose(s.bodies[law.a], q), b = body_pose(s.bodies[law.b], q);
        int invalid = 0;
        for (int k = 0; k < cache.anchors; ++k)
            if (!(transport(cache.anchor[k], a, b, invalid).depth > 0) || invalid)
                return false;
    }
    return true;
}
enum NewtonDispatch { reject_step, accept_step, refresh_contact };

// Run the small dense solve on one lane. Yield the exact pending line-search
// trial to the geometry grid whenever any of its witnesses has expired.
__device__ NewtonDispatch correct_step(const DeviceScene &s, double dt, int &iter, double &fraction,
                                       bool evaluated) {
    auto &c = *s.control;
    auto &trial = c.working;
    for (;;) {
        if (evaluated) {
            if (c.next.error < trial.error) {
                trial = c.next;
                ++iter;
                fraction = 0;
            } else {
                fraction *= .5;
                if (fraction < 1. / 128)
                    return reject_step;
            }
        }
        if (!fraction) {
            double f = 0;
            for (int j = 0; j < s.n; ++j)
                f = fmax(f, fmax(fabs(c.state.force[j]), fabs(trial.state.force[j])));
            if (isfinite(trial.error) && trial.error <= 1e-9 + 1e-8 * f) {
                c.state = trial.state;
                s.workspace->maximum_residual = fmax(s.workspace->maximum_residual, trial.error);
                return accept_step;
            }
            if (iter == 18 || !isfinite(trial.error) || !cholesky(trial, s.n, c.delta))
                return reject_step;
            fraction = 1;
        }
        c.next.state.time = trial.state.time;
        for (int j = 0; j < s.n; ++j) {
            const double a = trial.state.a[j] + fraction * c.delta[j];
            c.next.state.a[j] = a;
            c.next.state.q[j] = c.qpred[j] + dt * dt * a;
            c.next.state.v[j] = c.vpred[j] + 1.5 * dt * a;
        }
        if (!anchors_cover(s, c.next.state.q))
            return refresh_contact;
        forces(ScalarGrid{}, s, c.next, dt, false);
        if (c.status)
            return reject_step;
        evaluated = true;
    }
}
template <class Grid> __device__ bool solve_step(Grid grid, const DeviceScene &s, double dt) {
    auto &c = *s.control;
    auto &trial = c.working;
    int lane = grid.thread_rank();
    if (!lane) {
        for (int j = 0; j < s.np; ++j)
            s.cache[j].ready = 0;
        trial.state.time = c.state.time + dt;
        for (int i = 0; i < s.n; ++i) {
            c.qpred[i] = c.state.q[i] + dt * c.state.v[i] + dt * dt * (-.5) * c.state.a[i];
            c.vpred[i] = c.state.v[i] + dt * (-.5) * c.state.a[i];
            trial.state.a[i] = c.state.a[i];
            trial.state.q[i] = c.qpred[i] + dt * dt * trial.state.a[i];
            trial.state.v[i] = c.vpred[i] + 1.5 * dt * trial.state.a[i];
        }
    }
    grid.sync();
    forces(grid, s, trial, dt, false);
    if (c.status)
        return false;
    int iter = 0;
    double fraction = 0;
    bool evaluated = false;
    for (;;) {
        if (!lane)
            c.dispatch = correct_step(s, dt, iter, fraction, evaluated);
        grid.sync();
        const int dispatch = c.dispatch;
        grid.sync();
        if (dispatch != refresh_contact)
            return dispatch == accept_step;
        forces(grid, s, c.next, dt, false);
        if (c.status)
            return false;
        evaluated = true;
    }
}
__device__ void save_frame(const DeviceScene &s, int step) {
    auto &z = s.control->state;
    int width = 1 + 4 * s.n + 3 * s.np;
    double *f = s.frames + static_cast<size_t>(step) * width;
    *f++ = z.time;
    for (int i = 0; i < s.n; ++i) {
        *f++ = z.q[i];
        *f++ = z.v[i];
        *f++ = z.a[i];
        *f++ = z.force[i];
    }
    for (int i = 0; i < 3 * s.np; ++i)
        *f++ = s.observed[i];
}
template <class Grid> __device__ void advance_steps(Grid grid, DeviceScene s, int begin, int end, double dt) {
    int lane = grid.thread_rank();
    auto &c = *s.control;
    if (c.status)
        return;
    if (begin == 0) {
        if (!lane)
            s.finished[0] = timer_ns();
        if (!lane)
            for (int i = 0; i < s.n; ++i) {
                c.working.state.q[i] = s.coordinates[i].q;
                c.working.state.v[i] = s.coordinates[i].v;
            }
        grid.sync();
        forces(grid, s, c.working, dt, true);
        if (!lane && c.invalid)
            atomicCAS(&c.status, 0, 8);
        grid.sync();
        if (c.status)
            return;
        if (!lane) {
            c.state = c.working.state;
            for (int i = 0; i < s.n; ++i)
                c.state.a[i] = c.state.force[i] / s.coordinates[i].mass;
            save_frame(s, 0);
        }
        grid.sync();
    }
    for (int i = begin; i < end; ++i) {
        if (!lane)
            c.last_step = i;
        grid.sync();
        unsigned level = 0, cell = 0;
        for (;;) {
            bool ok = solve_step(grid, s, ldexp(dt, -static_cast<int>(level)));
            grid.sync();
            if (c.status)
                return;
            if (!ok) {
                if (level == 8) {
                    if (!lane)
                        c.status = 7;
                    return;
                }
                ++level;
                cell *= 2;
                if (!lane)
                    ++s.workspace->subdivisions;
            } else {
                ++cell;
                while (level && !(cell & 1)) {
                    cell /= 2;
                    --level;
                }
                if (!level)
                    break;
            }
        }
        if (!lane) {
            save_frame(s, i + 1);
            s.finished[i + 1] = timer_ns();
        }
        grid.sync();
    }
}
__global__ void resident_steps(DeviceScene s, int begin, int end, double dt) {
    advance_steps(cooperative_groups::this_grid(), s, begin, end, dt);
}
__global__ void unconstrained_steps(DeviceScene s, int begin, int end, double dt) {
    advance_steps(ScalarGrid{}, s, begin, end, dt);
}
