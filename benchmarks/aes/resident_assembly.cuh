// Compiled after resident_contact.cu. Each contacting body has one ground
// hinge/slider; several bodies can reference the same generalized coordinate.
constexpr int maxCoordinates = 8;
constexpr int ticksPerStep = ASSEMBLY_TICKS;
struct AssemblyState {
    double q[maxCoordinates], v[maxCoordinates], a[maxCoordinates], previous[maxCoordinates];
    double x[maxCoordinates], speed[maxCoordinates], trial[maxCoordinates], base[maxCoordinates],
        delta[maxCoordinates];
    double lastError, refineUntil;
    int step, tick, span, iter, halves, failed, attempts, accepted, previousContacts, impactSpan;
    int explicitStep, explicitCount;
    double terminalResidual, terminalFrequency;
    unsigned long long start;
};

__device__ void next_assembly_trial(AssemblyState &s, int n, double h) {
    double dt = h * s.span / ticksPerStep;
    for (int i = 0; i < n; ++i) {
        s.x[i] = s.q[i] + dt * s.v[i] + dt * dt * (.1975 * s.a[i] + .3025 * s.trial[i]);
        s.speed[i] = s.v[i] + dt * (.4 * s.a[i] + .6 * s.trial[i]);
        if (s.explicitStep) {
            s.x[i] = s.q[i] + dt * s.v[i] + .5 * dt * dt * s.base[i];
            s.speed[i] = s.v[i] + dt * s.base[i];
        }
    }
}
__device__ MeshView &mesh_at(unsigned char *views, int i) {
    return *reinterpret_cast<MeshView *>(views + 320 * i);
}
// Each trial needs two independently transformed source bounds.
__device__ void assembly_frame(State &f, const Joint *j, const int *coords, const AssemblyState &s, double t,
                               const MeshView &mesh, const double *dirs, const int *bins, double radius) {
    f.pos[0] = s.x[coords[0]];
    f.pos[1] = s.x[coords[1]];
    set_pose(f, j, t);
    f.cold = 1;
    source_interval(f, mesh.bounds, dirs, bins, mesh.n, radius, 0.);
}
struct SearchOrder {
    const double *dirs;
    const int *bins, *indices;
    double radius;
};
extern "C" __global__ void bind_order(SearchOrder *out, const double *d, const int *b, const int *i,
                                      double r) {
    static_assert(sizeof(SearchOrder) == 32);
    *out = {d, b, i, r};
}
extern "C" __global__ void assembly_start(AssemblyState *s, const double *coord, int n) {
    static_assert(sizeof(AssemblyState) <= 1024);
    static_assert(sizeof(State) <= 1024);
    for (int i = 0; i < n; ++i)
        s->x[i] = s->q[i] = coord[4 * i + 3];
}
extern "C" __global__ void
assembly_advance(AssemblyState *state, unsigned char *views, State *frames, const SearchOrder *orders,
                 const Joint *joints, const int *map, const int *prefer, const double *coord,
                 const double *stiff, const double *loads, int n, int pairs, double h, int stop,
                 int contactSamples, int allowExplicit, double *trace, unsigned long long *elapsed
#ifdef NC_ASSEMBLY_AUDIT
                 ,
                 int auditFirst, int auditLast, double *audit, int *auditCount, int auditCapacity,
                 int regionStep, double *regionAudit, int *regionCount, int regionCapacity
#endif
) {
    auto all = cg::this_grid();
    int worker = blockIdx.x * blockDim.x + threadIdx.x;
    AssemblyState &s = *state;
    while (s.step < stop && !s.failed) {
        if (!worker) {
            if (s.step && s.tick == 0 && !s.iter && !s.halves && !s.start)
                s.start = gpu_time();
            next_assembly_trial(s, n, h);
            double t = s.step ? (s.step - 1 + double(s.tick + s.span) / ticksPerStep) * h : 0.;
            for (int d = 0; d < 2 * pairs; ++d)
                assembly_frame(frames[d], joints + 2 * d, map + 2 * d, s, t, mesh_at(views, d),
                               orders[d].dirs, orders[d].bins, orders[d].radius);
        }
        all.sync();
        for (int d = 0; d < 2 * pairs; ++d) {
            MeshView &m = mesh_at(views, d);
            const State &f = frames[d];
            for (int k = worker / QUERY_LANES; k < f.probeCount; k += gridDim.x * blockDim.x / QUERY_LANES) {
                int src = orders[d].indices[f.firstProbe + k];
                sample_contact<QUERY_LANES>(src, f.pose, VIEW(m));
            }
        }
        all.sync();
        if (!worker) {
            ++s.attempts;
            double K[maxCoordinates][maxCoordinates] = {}, applied[maxCoordinates], err = 0, scale = 0,
                   frequency2 = 0;
            int active = 0;
            double frac = double(s.tick + s.span) / ticksPerStep;
            double *out = trace + s.step * (2 * n + 4 * pairs);
            for (int i = 0; i < n; ++i) {
                double load = loads[s.step * n + i];
                if (s.step && frac < 1.)
                    load = loads[(s.step - 1) * n + i] * (1. - frac) + frac * load;
                applied[i] = load - coord[4 * i + 1] * s.x[i] - coord[4 * i + 2] * s.speed[i];
            }
            for (int p = 0; p < pairs; ++p) {
                MeshView &m = mesh_at(views, 2 * p), &back = mesh_at(views, 2 * p + 1);
                State &f = frames[2 * p];
                merge_reverse(back, f.pose, m.n, prefer[p], m.keys, m.peaks, m.mask, m.slots, m.count,
                              m.witness);
                int a = map[4 * p], b = map[4 * p + 1], count = *m.count;
                for (int i = 1; i < count; ++i) {
                    int slot = m.slots[i], j = i;
                    while (j && m.keys[m.slots[j - 1]] > m.keys[slot]) {
                        m.slots[j] = m.slots[j - 1];
                        --j;
                    }
                    m.slots[j] = slot;
                }
                int bearing = count;
#ifdef NC_PAIR_STIFFNESS
                // One spring for the pair: U = k/2 * max_i(depth_i)^2.
                // The sorted region key breaks exact ties at a change of active witness.
                if (count) {
                    int deepest = m.slots[0];
                    for (int i = 1; i < count; ++i) {
                        int slot = m.slots[i];
                        if (m.witness[8 * (unsigned)m.peaks[slot]] >
                            m.witness[8 * (unsigned)m.peaks[deepest]])
                            deepest = slot;
                    }
                    for (int i = 0; i < count; ++i) {
                        int slot = m.slots[i];
                        if (slot != deepest) {
                            m.keys[slot] = vacant;
                            m.peaks[slot] = 0;
                        }
                    }
                    m.slots[0] = deepest;
                    bearing = 1;
                }
#endif
                active += bearing;
                double total = 0, ta = 0, tb = 0;
                for (int i = 0; i < bearing; ++i) {
                    int slot = m.slots[i];
                    const double *w = m.witness + 8 * (unsigned)m.peaks[slot];
                    P normal = from_a(f, {w[4], w[5], w[6]}), ra = from_a(f, {w[1], w[2], w[3]});
                    P rb = {ra.x + f.origin[0] - f.origin[3], ra.y + f.origin[1] - f.origin[4],
                            ra.z + f.origin[2] - f.origin[5]};
                    double ja = -arm(joints[4 * p], normal, ra), jb = arm(joints[4 * p + 1], normal, rb);
                    double force = .001 * stiff[p] * w[0];
#ifdef NC_ASSEMBLY_AUDIT
                    if (s.step == regionStep) {
                        int row = (*regionCount)++;
                        if (row < regionCapacity) {
                            double *r = regionAudit + row * 14;
                            r[0] = *auditCount;
                            r[1] = p;
                            r[2] = m.keys[slot] >> 32;
                            r[3] = (unsigned)m.keys[slot];
                            r[4] = (unsigned)m.peaks[slot];
                            for (int k = 0; k < 7; ++k)
                                r[5 + k] = w[k];
                            r[12] = ja * force;
                            r[13] = jb * force;
                        }
                    }
#endif
                    total += force;
                    ta += ja * force;
                    tb += jb * force;
                    K[a][a] += stiff[p] * ja * ja;
                    K[b][b] += stiff[p] * jb * jb;
                    K[a][b] += stiff[p] * ja * jb;
                    K[b][a] += stiff[p] * ja * jb;
                    m.keys[slot] = vacant;
                    m.peaks[slot] = 0;
                }
                *m.count = 0;
                applied[a] += ta;
                applied[b] += tb;
                out[2 * n + 4 * p] = total;
                out[2 * n + 4 * p + 1] = ta;
                out[2 * n + 4 * p + 2] = tb;
                out[2 * n + 4 * p + 3] = bearing;
            }
            double residual[maxCoordinates];
            for (int i = 0; i < n; ++i) {
                residual[i] = coord[4 * i] * s.trial[i] - .9 * applied[i] - .1 * s.previous[i];
                err = isfinite(residual[i]) ? fmax(err, fabs(residual[i]))
                                            : __longlong_as_double(0x7ff0000000000000LL);
                scale = fmax(scale, fmax(fabs(applied[i]), fabs(s.previous[i])));
                frequency2 += K[i][i] / coord[4 * i];
            }
            int want = ticksPerStep;
            if (s.step && active && (contactSamples || !s.previousContacts) && frequency2 > 0.) {
                double half = 3.141592653589793 / sqrt(frequency2);
                int samples = contactSamples ? contactSamples : 32;
                while (want > 1 && samples * h * want / ticksPerStep > half)
                    want /= 2;
                if (!contactSamples && want < s.span) {
                    s.refineUntil =
                        fmax(s.refineUntil, (s.step - 1 + double(s.tick) / ticksPerStep) * h + half);
                    s.impactSpan = want;
                }
            }
#ifdef NC_ASSEMBLY_AUDIT
            double *record = nullptr;
            int oldStep = s.step, oldTick = s.tick, oldSpan = s.span, oldAccepted = s.accepted;
            if (s.step >= auditFirst && s.step <= auditLast) {
                int index = (*auditCount)++;
                if (index < auditCapacity) {
                    record = audit + index * (8 + 5 * n + 4 * pairs);
                    record[0] = s.step;
                    record[1] = s.tick;
                    record[2] = s.span;
                    record[3] = s.iter;
                    record[4] = s.halves;
                    record[5] = err;
                    record[6] = scale;
                    for (int i = 0; i < n; ++i) {
                        record[8 + i] = s.x[i];
                        record[8 + n + i] = s.speed[i];
                        record[8 + 2 * n + i] = s.a[i];
                        record[8 + 3 * n + i] = s.trial[i];
                        record[8 + 4 * n + i] = applied[i];
                    }
                    for (int i = 0; i < 4 * pairs; ++i)
                        record[8 + 5 * n + i] = out[2 * n + i];
                }
            }
#endif
            if (s.explicitStep &&
                (!isfinite(err) || !isfinite(frequency2) || (h / ticksPerStep) * sqrt(frequency2) >= .01)) {
                s.failed = 2;
            } else if (s.step && want < s.span) {
                s.span = want;
                s.iter = 0;
                s.halves = 0;
                for (int i = 0; i < n; ++i)
                    s.trial[i] = s.a[i];
            } else if (!s.step || (isfinite(err) && (s.explicitStep || err <= 1e-7 * (1 + scale)))) {
                ++s.accepted;
                s.previousContacts = active;
                for (int i = 0; i < n; ++i) {
                    if (s.explicitStep) {
                        s.trial[i] = applied[i] / coord[4 * i];
                        s.speed[i] = s.v[i] + .5 * (h / ticksPerStep) * (s.base[i] + s.trial[i]);
                    }
                    s.q[i] = s.x[i];
                    s.v[i] = s.speed[i];
                    s.a[i] = s.step ? s.trial[i] : applied[i] / coord[4 * i];
                    s.trial[i] = s.a[i];
                    s.previous[i] = applied[i];
                }
                s.tick += s.span;
                s.explicitCount += s.explicitStep;
                s.explicitStep = 0;
                if (!s.step || s.tick == ticksPerStep) {
                    for (int i = 0; i < n; ++i) {
                        out[i] = s.q[i];
                        out[n + i] = s.v[i];
                    }
                    elapsed[s.step] = s.step ? gpu_time() - s.start : 0;
                    ++s.step;
                    s.tick = 0;
                    s.start = 0;
                    s.span = contactSamples
                                 ? want
                                 : ((s.step - 1) * h < s.refineUntil ? s.impactSpan : ticksPerStep);
                } else
                    s.span = min(ticksPerStep == 64 ? s.span : min(want, 2 * s.span), ticksPerStep - s.tick);
                s.iter = 0;
                s.halves = 0;
            } else if (!isfinite(err) || s.iter > 20 || s.halves > 10) {
                s.terminalResidual = err;
                s.terminalFrequency = frequency2;
                if (s.span == 1 && allowExplicit && !s.explicitStep && isfinite(err) &&
                    (h / ticksPerStep) * sqrt(frequency2) < .01) {
                    // A discontinuous witness can leave no HHT root across a switch.
                    // Redo only this minimum interval with a velocity-Verlet predictor/corrector.
                    s.explicitStep = 1;
                    for (int i = 0; i < n; ++i)
                        s.base[i] = s.previous[i] / coord[4 * i];
                } else if (s.span <= 1)
                    s.failed = 1;
                else {
                    s.span /= 2;
                    s.iter = 0;
                    s.halves = 0;
                    for (int i = 0; i < n; ++i)
                        s.trial[i] = s.a[i];
                }
            } else if (s.iter && err >= s.lastError) {
                ++s.halves;
                for (int i = 0; i < n; ++i)
                    s.trial[i] = s.base[i] + ldexp(s.delta[i], -s.halves);
            } else {
                // SPD effective tangent of the shared-coordinate HHT equations.
                double dt = h * s.span / ticksPerStep, factor = .9 * .3025 * dt * dt;
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j <= i; ++j)
                        K[i][j] *= factor;
                    K[i][i] += coord[4 * i] + .54 * dt * coord[4 * i + 2] + factor * coord[4 * i + 1];
                    for (int j = 0; j <= i; ++j) {
                        double a = K[i][j];
                        for (int k = 0; k < j; ++k)
                            a -= K[i][k] * K[j][k];
                        K[i][j] = (i == j) ? sqrt(a) : a / K[j][j];
                    }
                    double v = -residual[i];
                    for (int k = 0; k < i; ++k)
                        v -= K[i][k] * s.delta[k];
                    s.delta[i] = v / K[i][i];
                }
                for (int i = n - 1; i >= 0; --i) {
                    for (int k = i + 1; k < n; ++k)
                        s.delta[i] -= K[k][i] * s.delta[k];
                    s.delta[i] /= K[i][i];
                    s.base[i] = s.trial[i];
                    s.trial[i] += s.delta[i];
                }
                s.lastError = err;
                ++s.iter;
                s.halves = 0;
            }
#ifdef NC_ASSEMBLY_AUDIT
            if (record) {
                // 0 accepted, 1 stiffness refinement, 2 convergence subdivision,
                // 3 line search, 4 Newton update, 5 terminal failure, 6 explicit retry.
                record[7] = s.accepted != oldAccepted ? 0
                            : s.failed                ? 5
                            : s.span < oldSpan && oldStep == s.step && oldTick == s.tick
                                ? (want < oldSpan ? 1 : 2)
                            : s.explicitStep ? 6
                            : s.halves       ? 3
                                             : 4;
            }
#endif
        }
        all.sync();
    }
}
extern "C" __global__ void assembly_status(const AssemblyState *s, int *o) {
    o[0] = s->step;
    o[1] = s->failed;
    o[2] = s->attempts;
    o[3] = s->accepted;
    o[4] = s->explicitCount;
}
extern "C" __global__ void assembly_failure(const AssemblyState *s, double *o) {
    o[0] = s->failed;
    o[1] = s->tick;
    o[2] = s->span;
    o[3] = s->iter;
    o[4] = s->halves;
    o[5] = s->terminalResidual;
    o[6] = s->terminalFrequency;
    o[7] = s->explicitStep;
}
