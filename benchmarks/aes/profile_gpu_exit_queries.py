"""Compare short material-exit rays against the two surface representations."""

import argparse
import json
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import cupy as cp
import meshio
import numpy as np

from profile_gpu_mesh_queries import read_poses
from nurbscontact_gpu.bvh import IntersectionQuery, TriangleMesh


# Distances use the same millimetre convention as the compiled CAD kernel.
MESH_RAYS = r"""
struct ExitSelection {
    double best;
    int face;
    V normal;
    bool ambiguous;
};
__device__ void test_facet(const V* t,int id,V o,V d,double reach,ExitSelection &s) {
    const V e=minus_v(t[1],t[0]),f=minus_v(t[2],t[0]),p=cross_v(d,f);
    const double det=proj(e,p);
    if(det==0) return;
    const V delta=minus_v(o,t[0]),q=cross_v(delta,e);
    const double u=proj(delta,p)/det,v=proj(d,q)/det,h=proj(f,q)/det;
    if(u<0||v<0||u+v>1||h < -1e-7 || h>reach || h>s.best+1e-7) return;
    V nn=cross_v(e,f);
    nn=times_v(nn,1/norm_v(nn));
    if(s.face<0||h<s.best-1e-7) s={h,id,nn,false};
    else if(proj(nn,s.normal)<1-1e-9) s.ambiguous=true;
}
extern "C" __global__ void mesh_exits(const V* tris,const int* ids,const int* links,const double* boxes,
        int nodes,const double* rays,int n,double reach,double* result,const double* origin,
        const int* size,const int* offsets,const int* candidates) {
    const int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n) return;
    const double* ray=rays+6*i;
    const V o=make_double3(ray[0],ray[1],ray[2]),d=make_double3(ray[3],ray[4],ray[5]);
    ExitSelection s={reach+1e-7,-1,{},false};
    if(size[0]){
        int cell=0;
        bool inside=true;
        for(int k=0;k<3;++k){
            double x=floor((ray[k]-origin[k])*origin[3]);
            if(x<0||x>=size[k]){inside=false;break;}
            cell=cell*size[k]+int(x);
        }
        if(inside) for(int k=offsets[cell];k<offsets[cell+1];++k) {
            int j=candidates[k];
            test_facet(tris+3*j,ids[j],o,d,reach,s);
        }
    } else for(int b=0;b<nodes;){
        const double* box=boxes+6*b;
        const int* link=links+3*b;
        double low=-1e-7,high=s.best+1e-7;
        for(int k=0;k<3;++k){
            double p=box[k]-ray[k]-1e-7,q=box[3+k]-ray[k]+1e-7,v=ray[3+k];
            if(v==0) {if(p>0||q<0)high=-1;continue;}
            low=fmax(low,fmin(p/v,q/v));high=fmin(high,fmax(p/v,q/v));
        }
        if(low>high){b=link[0];continue;}
        for(int j=link[1];j<link[1]+link[2];++j) test_facet(tris+3*j,ids[j],o,d,reach,s);
        ++b;
    }
    if(s.face>=0 && (s.ambiguous||s.best<=1e-7||proj(s.normal,d)<=1e-8)) s.face=-1;
    result[2*i]=s.face;result[2*i+1]=s.best;
}
"""


def ray_grid(tri, reach):
    # Match the bounded-ray binning policy of compiled_cuda_pack.cpp.
    pad = 1e-6
    low, high = tri.min(axis=(0, 1)), tri.max(axis=(0, 1))
    width = max(reach, (high - low).max() / 160, 8 * pad)
    origin = low - (reach + pad)
    size = np.maximum(1, np.ceil((high + reach + pad - origin) / width)).astype(
        np.int32
    )
    cells = int(np.prod(size, dtype=np.int64))
    lo = np.clip(
        np.floor((tri.min(axis=1) - reach - pad - origin) / width), 0, size - 1
    ).astype(int)
    hi = np.clip(
        np.floor((tri.max(axis=1) + reach + pad - origin) / width), 0, size - 1
    ).astype(int)
    spans = hi - lo + 1
    count = np.prod(spans, axis=1)
    if cells > 3000000 or count.sum() > 16000000:
        return None
    owners = np.repeat(np.arange(len(tri), dtype=np.int32), count)
    local = np.arange(len(owners)) - np.repeat(np.cumsum(count) - count, count)
    z = local % spans[owners, 2] + lo[owners, 2]
    plane = local // spans[owners, 2]
    y = plane % spans[owners, 1] + lo[owners, 1]
    x = plane // spans[owners, 1] + lo[owners, 0]
    keys = (x * size[1] + y) * size[2] + z
    offsets = np.r_[0, np.cumsum(np.bincount(keys, minlength=cells))].astype(np.int32)
    return (
        np.r_[origin, 1 / width],
        size,
        offsets,
        owners[np.argsort(keys, kind="stable")],
    )


def ray_sets(scene, trace, samples, reach):
    files, pairs, poses, times = read_poses(scene, trace)
    meshes = []
    for f in files:
        mesh = meshio.read(f)
        meshes.append(
            TriangleMesh(
                mesh.points * 1000, mesh.get_cells_type("triangle"), leaf_size=4
            )
        )
    rays = [[] for _ in files]
    for pair, (a, b) in enumerate(pairs):
        query = IntersectionQuery(meshes[a], meshes[b])
        for index in np.unique(np.linspace(0, len(times) - 2, samples, dtype=int)):
            pose = poses[pair][index].copy()
            pose[:, 3] *= 1000
            sizes, vertices, normals = query.enqueue(pose, geometry=True)
            count = len(query.read())
            if not count:
                continue
            transverse = sizes[:count].get() == 2
            center = vertices[:count, :2].get()[transverse].mean(axis=1)
            normals = normals[:count].get()[transverse]
            pose = pose.get()
            for side, body in enumerate((a, b)):
                normal = normals[:, side]
                point = center
                if not side:
                    normal = normal @ pose[:, :3]
                    point = (point - pose[:, 3]) @ pose[:, :3]
                for sign in (-1, 1):
                    rays[body].append(
                        np.column_stack((point + sign * 0.25 * reach * normal, normal))
                    )
    ordered = []
    for rows in rays:
        # Make ordering within each pose independent of device atomics while
        # keeping the sampled poses in their original temporal order.
        # Each entry is an independent set from one pose and one offset side.
        ordered.append(
            np.ascontiguousarray(
                np.concatenate([row[np.lexsort(row.T[::-1])] for row in rows])
            )
        )
    return files, meshes, ordered


def mesh_worker(argv, index):
    mesh_file, ray_file, reach, repeats, out_file, batch = argv
    reach, repeats, batch = float(reach), int(repeats), int(batch)
    model = meshio.read(mesh_file)
    mesh = TriangleMesh(
        model.points * 1000, model.get_cells_type("triangle"), leaf_size=4
    )
    grid = ray_grid(mesh.triangles.get(), reach) if index == "grid" else None
    if index == "grid" and grid is None:
        raise ValueError("mesh bounded-ray grid exceeds the common incidence budget")
    if grid is None:
        grid = (np.zeros(4), *[np.zeros(3, dtype=np.int32) for _ in range(3)])
    grid = tuple(cp.asarray(x) for x in grid)
    src = (
        Path(__file__).resolve().parents[2]
        / "nurbscontact_gpu/kernels/gpu_triangle_intersections.cu"
    )
    kernel = cp.RawModule(
        code=src.read_text() + MESH_RAYS, options=("--std=c++14", "--fmad=false")
    ).get_function("mesh_exits")
    rays = cp.asarray(np.fromfile(ray_file, dtype=np.float64).reshape(-1, 6))
    hits = cp.empty((len(rays), 2), cp.float64)
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()
    launches = (len(rays) + batch - 1) // batch
    timings = []
    with stream:
        stream.begin_capture()
        for i in range(0, len(rays), batch):
            n = min(batch, len(rays) - i)
            kernel(
                ((n + 63) // 64,),
                (64,),
                (
                    mesh.triangles,
                    mesh.ids,
                    mesh.links,
                    mesh.bounds,
                    np.int32(len(mesh.links)),
                    rays[i:],
                    np.int32(n),
                    np.float64(reach),
                    hits[i:],
                    *grid,
                ),
            )
        graph = stream.end_capture()
        start, end = cp.cuda.Event(), cp.cuda.Event()
        for _ in range(3):
            until = time.perf_counter() + 0.25
            while time.perf_counter() < until:
                graph.launch(stream)
                stream.synchronize()
            start.record(stream)
            for _ in range(repeats):
                graph.launch(stream)
            end.record(stream)
            end.synchronize()
            timings.append(
                1000 * cp.cuda.get_elapsed_time(start, end) / (launches * repeats)
            )
    print(json.dumps(timings))
    hits.get().tofile(out_file)


def run(args):
    files, _, rays = ray_sets(args.scene, args.trace, args.poses, args.reach)
    args.output.mkdir(exist_ok=True, parents=True)
    report = []
    for f, arr in zip(files, rays):
        ray_file = args.output / (f.stem + ".bin")
        arr.tofile(ray_file)
        results = {}
        means = {}
        launches = (len(arr) + args.batch - 1) // args.batch
        timings = {}
        for name, command, part in (
            ("cad", [str(args.exe)], f.with_suffix(".ncpart")),
            (
                "mesh",
                [sys.executable, str(Path(__file__).resolve()), "--mesh-worker", "bvh"],
                f,
            ),
            (
                "mesh_grid",
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--mesh-worker",
                    "grid",
                ],
                f,
            ),
        ):
            output = args.output / (f.stem + "-" + name + ".exits")
            result = subprocess.run(
                [
                    *command,
                    str(part),
                    str(ray_file),
                    str(args.reach),
                    str(args.repeats),
                    str(output),
                    str(args.batch),
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                timeout=1800,
            )
            timings[name] = json.loads(result.stdout.strip().splitlines()[-1])
            if len(timings[name]) != 3:
                raise ValueError("three independent graph timing trials required")
            means[name] = float(np.median(timings[name]))
            results[name] = np.fromfile(output, dtype=np.float64).reshape(-1, 2)
            if len(results[name]) != len(arr):
                raise ValueError("incomplete exit results")
        cad, mesh = results["cad"], results["mesh"]
        grid = results["mesh_grid"]
        grid_status_diff = int(np.count_nonzero((mesh[:, 0] >= 0) != (grid[:, 0] >= 0)))
        both = (mesh[:, 0] >= 0) & (grid[:, 0] >= 0)
        grid_distance_diff = float(
            np.max(np.abs(mesh[both, 1] - grid[both, 1]), initial=0)
        )
        if grid_status_diff or grid_distance_diff > 1e-7:
            raise ValueError("mesh spatial indices disagree on material exits")
        common = (cad[:, 0] >= 0) & (mesh[:, 0] >= 0)
        err = np.abs(cad[common, 1] - mesh[common, 1])
        report.append(
            dict(
                part=f.stem,
                rays=len(arr),
                max_batch=args.batch,
                launches_per_repeat=launches,
                median_mean_batch_us=means,
                trials_mean_batch_us=timings,
                max_ray_mm=args.reach,
                graph_repeats=args.repeats,
                poses=args.poses,
                ray_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
                mesh_indices_status_differences=grid_status_diff,
                mesh_indices_max_distance_difference_mm=grid_distance_diff,
                cad_hits=int(sum(cad[:, 0] >= 0)),
                mesh_hits=int(sum(mesh[:, 0] >= 0)),
                both_hits=int(sum(common)),
                different_hit_status=int(sum((cad[:, 0] >= 0) != (mesh[:, 0] >= 0))),
                cad_unresolved=int(sum(cad[:, 0] == -2)),
                common_median_distance_um=(
                    float(np.median(err) * 1000) if len(err) else None
                ),
            )
        )
        (args.output / "comparison.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print(json.dumps(report))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--mesh-worker"]:
        mesh_worker(sys.argv[3:], sys.argv[2])
        sys.exit(0)
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "trace", "exe", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--poses", type=int, default=41)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument(
        "--reach", type=float, required=True, help="ray length in millimetres"
    )
    run(p.parse_args())
