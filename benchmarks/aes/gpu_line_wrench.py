"""Replay the local fitted-CAD force path on the archived line quadrature.

Each input line supplies its two face IDs, world-space quadrature points on B
(in A's frame) and weights. The GPU projects onto A's fitted face, then sums
the force and moment on B. The CPU checks the returned UV against the STEP
trim and compares gaps and line wrenches with the independent OCCT replay.
With --mesh-um the mesh projects the same points onto triangles of the same
STEP face. With --full-mesh it searches the complete rigid mesh of A.
All methods use the same quadrature and force law. Region
discovery and HHT integration are outside this timing.

From the repository root, after the cold region replay:
    python benchmarks/aes/gpu_line_wrench.py results/gears/models/spur-1a.step results/aes/gear-field/faces.npz results/aes/gear-field/transition-points.csv build/aes-region-replay/spur-poses.txt build/aes-region-replay/cold.loads.csv results/aes/gpu-face-candidates/spur-wrenches.json --cutoff-mm 0.055 --stiffness 1e9
"""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import cupy as cp
import numpy as np

from gpu_face_candidates import resident_screen_us
from nhrep_gear import read_occt_solid


POINT_WRENCH = r'''
__device__ void add_point_wrench(const double* p, const double* center, const double* d,
                                 double gap, double cutoff_mm, double stiffness, double* w) {
    if (!(gap>0)) { for (int k=0;k<6;++k) w[k]=nan(""); return; }
    double scale=stiffness*p[3]*0.001*fmax(cutoff_mm-gap,0.0)/gap;
    double f[3], r[3];
    for (int k=0;k<3;++k) {
        f[k]=scale*d[k]; w[k]+=f[k];
        r[k]=p[k]*0.001-center[k];
    }
    w[3]+=r[1]*f[2]-r[2]*f[1];
    w[4]+=r[2]*f[0]-r[0]*f[2];
    w[5]+=r[0]*f[1]-r[1]*f[0];
}
'''

CUDA = POINT_WRENCH + r'''
// Evaluate the normalized Chebyshev chart and both first derivatives.
__device__ void surface(const double* coef, int nu, int nv, double u, double v,
                        double* x, double* xu, double* xv) {
    for (int k=0;k<3;++k) x[k]=xu[k]=xv[k]=0;
    // T[-1](x) = T[1](x) initializes the same recurrence at degree zero.
    double tu=1, prev_u=u, du=0, prev_du=1;
    for (int i=0;i<nu;++i) {
        double tv=1, prev_v=v, dv=0, prev_dv=1;
        for (int j=0;j<nv;++j) {
            for (int k=0;k<3;++k) {
                double c=coef[3*(i*nv+j)+k];
                x[k]+=tu*tv*c; xu[k]+=du*tv*c; xv[k]+=tu*dv*c;
            }
            double next=2*v*tv-prev_v;
            double deriv=2*(tv+v*dv)-prev_dv;
            prev_v=tv; tv=next; prev_dv=dv; dv=deriv;
        }
        double next=2*u*tu-prev_u;
        double deriv=2*(tu+u*du)-prev_du;
        prev_u=tu; tu=next; prev_du=du; du=deriv;
    }
}
__device__ double dot(const double* a, const double* b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
__device__ void basis(int degree, double s, double& value, double& deriv) {
    value=1; deriv=0;
    double last=s, dlast=1;
    for (int k=0;k<degree;++k) {
        double next=2*s*value-last, dnext=2*(value+s*deriv)-dlast;
        last=value; value=next; dlast=deriv; deriv=dnext;
    }
}
__device__ double warp_sum(double x) {
    for (int delta=16;delta>0;delta/=2) x+=__shfl_down_sync(0xffffffff,x,delta);
    return __shfl_sync(0xffffffff,x,0);
}
__device__ void warp_surface(const double* coef, int nu, int nv, double u, double v,
                             double* x, double* xu, double* xv) {
    for (int k=0;k<3;++k) x[k]=xu[k]=xv[k]=0;
    for (int q=threadIdx.x%32;q<nu*nv;q+=32) {
        double tu,du,tv,dv;
        basis(q/nv,u,tu,du); basis(q%nv,v,tv,dv);
        for (int k=0;k<3;++k) {
            double c=coef[3*q+k];
            x[k]+=tu*tv*c; xu[k]+=du*tv*c; xv[k]+=tu*dv*c;
        }
    }
    for (int k=0;k<3;++k) {
        x[k]=warp_sum(x[k]); xu[k]=warp_sum(xu[k]); xv[k]=warp_sum(xv[k]);
    }
}
template<bool warp>
__device__ bool project(const double* p, const double* coef, int nu, int nv,
                        double u, double v, double* delta, double* uv) {
    for (int iter=0;iter<16;++iter) {
        double x[3],xu[3],xv[3];
        if (warp) warp_surface(coef,nu,nv,u,v,x,xu,xv);
        else surface(coef,nu,nv,u,v,x,xu,xv);
        for (int k=0;k<3;++k) delta[k]=p[k]-x[k];
        double a=dot(xu,xu), b=dot(xu,xv), c=dot(xv,xv);
        double det=a*c-b*b;
        if (!(det>1e-14*(a+c)*(a+c))) return false;
        double ru=dot(xu,delta), rv=dot(xv,delta);
        double su=(c*ru-b*rv)/det, sv=(a*rv-b*ru)/det;
        if (a*su*su+2*b*su*sv+c*sv*sv < 1e-20) {
            uv[0]=u; uv[1]=v;
            return isfinite(u) && isfinite(v);
        }
        u+=su; v+=sv;
        if (!isfinite(u) || !isfinite(v)) return false;
    }
    return false;
}
template<bool warp>
__device__ void accumulate(int j, const int* chart, const double* coeffs, const double* samples,
                           const double* center, double cutoff_mm, double stiffness,
                           const int* prior, double* projections, double* w) {
    const double* p=samples+4*j;
    double* result=projections+3*j;
    double d[3], uv[2];
    int previous=prior ? prior[j] : -1;
    const double* c=coeffs+3*chart[0];
    bool ok=false;
    if (previous>=0) {
        const double* seed=projections+3*previous;
        if (isfinite(seed[1]) && isfinite(seed[2]))
            ok=project<warp>(p,c,chart[1],chart[2],seed[1],seed[2],d,uv);
    }
    if (!ok) ok=project<warp>(p,c,chart[1],chart[2],0,0,d,uv);
    if (warp && threadIdx.x%32!=0) return;
    double gap=ok ? sqrt(dot(d,d)) : nan("");
    result[0]=gap;
    result[1]=ok ? uv[0] : nan(""); result[2]=ok ? uv[1] : nan("");
    add_point_wrench(p,center,d,gap,cutoff_mm,stiffness,w);
}
extern "C" __global__ void line_wrench(const int* range, const int* offsets,
    const int* charts, const double* coeffs, const double* samples, const double* centers,
    double cutoff_mm, double stiffness, double* projections, double* wrenches) {
    int line=range[0]+blockDim.x*blockIdx.x+threadIdx.x;
    if (line>=range[1]) return;
    double w[6]={};
    for (int j=offsets[line];j<offsets[line+1];++j)
        accumulate<false>(j,charts+3*line,coeffs,samples,centers+3*line,cutoff_mm,stiffness,nullptr,projections,w);
    for (int k=0;k<6;++k) wrenches[6*line+k]=w[k];
}
extern "C" __global__ void parallel_line(const int* range, const int* offsets,
    const int* charts, const double* coeffs, const double* samples, const double* centers,
    double cutoff_mm, double stiffness, const int* prior, double* projections, double* wrenches) {
    int line=range[0]+blockIdx.x;
    if (line>=range[1]) return;
    const int warp=threadIdx.x/32;
    double w[6]={};
    for (int j=offsets[line]+warp;j<offsets[line+1];j+=blockDim.x/32)
        accumulate<true>(j,charts+3*line,coeffs,samples,centers+3*line,cutoff_mm,stiffness,prior,projections,w);
    extern __shared__ double partial[];
    if (threadIdx.x%32==0) for (int k=0;k<6;++k) partial[6*warp+k]=w[k];
    __syncthreads();
    if (threadIdx.x<6) {
        double total=0;
        for (int i=0;i<blockDim.x/32;++i) total+=partial[6*i+threadIdx.x];
        wrenches[6*line+threadIdx.x]=total;
    }
}
'''


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def run(args):
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.TopAbs import TopAbs_FACE, TopAbs_IN
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Pnt2d

    poses = np.loadtxt(args.poses, ndmin=2)
    centers = {row[0]: row[1:4] for row in poses}
    groups = {}
    for row in read_csv(args.samples):
        pair = tuple(map(int, row['face_pair'].split('/')))
        groups.setdefault((float(row['time_s']), *pair), []).append(row)
    expected = read_csv(args.loads)
    references = {(float(r['time_s']), int(r['face_a']), int(r['face_b'])): r for r in expected}
    if (not groups or groups.keys() != references.keys() or len(centers) != len(poses)
            or len(expected) != len(references)):
        raise ValueError('Missing, duplicate or mismatched lines or poses')
    keys = sorted(groups)
    offsets, points, target_gaps, target_wrenches = [0], [], [], []
    indices, frames, chart_ranges, coefs = {}, {}, [], []
    with np.load(args.fits) as fits:
        for face in sorted({key[1] for key in keys}):
            coef = fits[f'coef_{face}']
            indices[face] = (sum(len(c) for c in coefs), *coef.shape[:2])
            frames[face] = fits[f'center_{face}'], fits[f'half_{face}']
            coefs.append(coef.reshape(-1, 3))
    for key in keys:
        for r in groups[key]:
            points.append([float(r[c]) for c in ('x_mm', 'y_mm', 'z_mm', 'weight_m')])
            target_gaps.append(float(r['cad_distance_mm']))
        offsets.append(len(points))
        chart_ranges.append(indices[key[1]])
        target_wrenches.append([float(references[key][f'{q}_b{d}_{unit}'])
                                 for q,unit in [('force','N'), ('moment','Nm')] for d in 'xyz'])
    if not np.isfinite(points).all() or np.any(np.array(points)[:, 3] <= 0):
        raise ValueError('Invalid quadrature coordinates or weights')
    line_times = np.array([key[0] for key in keys])
    if set(line_times) != set(centers) or np.any(np.diff(poses[:,0])<=0):
        raise ValueError('This line replay requires at least one recorded line per pose')
    ranges = [np.searchsorted(line_times, [t, np.nextafter(t, np.inf)]).astype(np.int32) for t in centers]
    # The pose kernels execute serially. A seed can only refer to the same
    # Gauss-point ordinal on the same face pair in the immediately preceding pose.
    # A break in the recorded time grid starts a fresh cold window.
    prior = np.full(len(points), -1, np.int32)
    by_key = {key: i for i,key in enumerate(keys)}
    for previous, current in zip(poses[:-1,0], poses[1:,0]):
        if current-previous > args.max_seed_age_s:
            continue
        for i in np.flatnonzero(line_times==current):
            j = by_key.get((previous, *keys[i][1:]))
            if j is not None and offsets[j+1]-offsets[j] == offsets[i+1]-offsets[i]:
                prior[offsets[i]:offsets[i+1]] = np.arange(offsets[j], offsets[j+1])
    cold_seed = cp.full(len(points), -1, dtype=cp.int32)
    warm_seed = cp.asarray(prior)
    parameters = (cp.asarray(offsets, dtype=cp.int32), cp.asarray(chart_ranges, dtype=cp.int32),
                  cp.asarray(np.concatenate(coefs)), cp.asarray(points), cp.asarray([centers[k[0]] for k in keys]),
                  np.float64(args.cutoff_mm), np.float64(args.stiffness))
    projection = cp.full((len(points), 3), np.nan)
    wrenches = cp.full((len(keys), 6), np.nan)
    max_lines = max(int(r[1]-r[0]) for r in ranges)
    serial = cp.RawKernel(CUDA, 'line_wrench')
    parallel = cp.RawKernel(CUDA, 'parallel_line')

    def launch_serial(interval):
        serial(((max_lines+31)//32,), (32,), (interval, *parameters, projection, wrenches))

    def launch_parallel(interval, seeds=cold_seed):
        parallel((max_lines,), (128,), (interval, *parameters, seeds, projection, wrenches), shared_mem=4*6*8)

    def launch_warm(interval):
        launch_parallel(interval, warm_seed)

    def evaluate(launch):
        projection.fill(np.nan)
        wrenches.fill(np.nan)
        for r in ranges:
            launch(cp.asarray(r))
        pr, wr = projection.get(), wrenches.get()
        if not np.isfinite(pr).all() or not np.isfinite(wr).all():
            raise ValueError('Unresolved local CAD projection')
        return pr, wr

    serial_pr, serial_wr = evaluate(launch_serial)
    cold_pr, cold_wr = evaluate(launch_parallel)
    pr, wr = evaluate(launch_warm)
    np.testing.assert_allclose(pr, cold_pr, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(wr, cold_wr, rtol=1e-8, atol=1e-7)
    np.testing.assert_allclose(pr, serial_pr, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(wr, serial_wr, rtol=1e-8, atol=1e-7)
    faces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(read_occt_solid(args.step), TopAbs_FACE, faces)
    for i, key in enumerate(keys):
        face = TopoDS.Face_s(faces.FindKey(key[1]))
        center, half = frames[key[1]]
        for uv in center + pr[offsets[i]:offsets[i+1], 1:] * half:
            if BRepClass_FaceClassifier(face, gp_Pnt2d(*uv), 1e-9).State() != TopAbs_IN:
                raise ValueError(f'Projected point outside the trimmed face interior at {key}')
    error = wr - np.array(target_wrenches)
    gaps_um = 1000*np.abs(pr[:, 0]-target_gaps)
    sources = (args.step, args.fits, args.samples, args.poses, args.loads)
    paired = []
    methods = [('serial', launch_serial), ('parallel', launch_parallel), ('previous_uv', launch_warm)]
    mesh_checks = {}
    if args.mesh_um or args.full_mesh:
        from gpu_mesh_line import MeshLines
        mesh_cases = [(f'mesh_{d:g}um',d,None) for d in (args.mesh_um or [])]
        mesh_cases += [(f'full_{p.stem}',None,p) for p in (args.full_mesh or [])]
        for label, deflection, path in mesh_cases:
            mesh = MeshLines(args.step, deflection, keys, max_lines, offsets, points,
                             [centers[k[0]] for k in keys], prior, args.cutoff_mm, args.stiffness,
                             POINT_WRENCH, path)
            mesh_checks[label] = mesh.check(ranges, target_gaps, target_wrenches)
            methods.append((label, mesh.launch))
    for i, group in enumerate(np.array_split(np.arange(args.repeats), min(5, args.repeats))):
        row = {'repeats': len(group)}
        start = i % len(methods)
        for name, launch in methods[start:] + methods[:start]:
            row[name] = resident_screen_us(launch, ranges, len(group))
        paired.append(row)
    report = dict(gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                input_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in sources},
                stage='known face and quadrature; fitted chart projection plus line force and moment on B',
                poses=len(poses), lines=len(keys), checked_trim_interior_points=len(pr), repeats=args.repeats,
                previous_pose_seed_available=int(np.count_nonzero(prior>=0)), max_seed_age_s=args.max_seed_age_s,
                cutoff_mm=args.cutoff_mm, stiffness_N_per_m2=args.stiffness,
                max_gap_error_um=float(max(gaps_um)), max_line_force_vector_error_N=float(max(np.linalg.norm(error[:, :3], axis=1))),
                max_line_moment_vector_error_Nm=float(max(np.linalg.norm(error[:, 3:], axis=1))),
                paired_resident_us_per_known_pose=paired,
                serial_resident_us_per_known_pose=float(np.median([row['serial'] for row in paired])),
                parallel_resident_us_per_known_pose=float(np.median([row['parallel'] for row in paired])),
                previous_uv_resident_us_per_known_pose=float(np.median([row['previous_uv'] for row in paired])))
    if mesh_checks:
        for label, check in mesh_checks.items():
            check['resident_us_per_known_pose'] = float(np.median([r[label] for r in paired]))
        report['triangle_meshes'] = mesh_checks
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('step','fits','samples','poses','loads','output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--cutoff-mm', type=float, required=True)
    parser.add_argument('--stiffness', type=float, required=True, help='Line stiffness in N/m²')
    parser.add_argument('--max-seed-age-s', type=float, default=25e-6,
                        help='Discard hints between poses separated by more than this time')
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--mesh-um', type=float, nargs='+', help='STEP triangle mesh deflections in micrometres')
    parser.add_argument('--full-mesh', type=Path, nargs='+', help='Search each complete v/f triangle archive in mm')
    args = parser.parse_args()
    if args.repeats < 1 or not all(np.isfinite(x) and x>0 for x in (args.cutoff_mm,args.stiffness,args.max_seed_age_s)):
        parser.error('Use finite, positive cutoff, stiffness and repeats')
    if args.mesh_um and (len(set(args.mesh_um)) != len(args.mesh_um)
                         or any(not np.isfinite(h) or h<=0 for h in args.mesh_um)):
        parser.error('Mesh deflections must be positive, finite and distinct')
    if args.full_mesh and len({p.stem for p in args.full_mesh}) != len(args.full_mesh):
        parser.error('Use distinct file stems for the full-mesh inputs')
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
