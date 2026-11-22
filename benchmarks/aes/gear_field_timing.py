"""Time warm signed point-to-solid queries for the STEP spur pinion on one CPU thread.

Run from the repository root with the OCCT DLL directories on PATH.
python benchmarks/aes/gear_field_timing.py build/aes-occt/Release/nc_contact.exe build/aes-gear-field-timing
"""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import subprocess

import numpy as np
from threadpoolctl import threadpool_limits


def summary(single, batch, pose, indices):
    mean_ms = np.median(batch)*1000/1536
    return {
        'single_median_ms':float(np.median(single)*1000),
        'single_p95_ms':float(np.percentile(single,95)*1000),
        'single_face_median_ms':float(np.median(single[:,indices<768])*1000),
        'single_edge_median_ms':float(np.median(single[:,indices>=768])*1000),
        'mixed_1536_seconds':batch,
        'mixed_1536_median_s':float(np.median(batch)),
        'mixed_per_point_ms':float(mean_ms),
        'mixed_points_per_second':float(1000/mean_ms),
        'four_contact_points_median_ms':float(np.median(pose)*1000),
    }


def accuracy(values, truth):
    err = 1000*abs(values-truth)
    if not np.isfinite(err).all():
        raise ValueError('Non-finite query result')
    return {'within_0p125_um':int(sum(err<=.125)), 'max_error_um':float(err.max())}


def profile_projections(field, points):
    """Measure time spent in the fitted surface projections on a separate pass."""
    project = field.project
    samples = []
    def timed(*args):
        start = perf_counter()
        result = project(*args)
        samples.append(perf_counter()-start)
        return result
    field.project = timed
    try:
        started = perf_counter()
        field.predict(points)
        elapsed = perf_counter()-started
    finally:
        field.project = project
    return {'points':len(points),'total_s':elapsed,'projection_s':sum(samples),
            'all_point_projection_s':samples[0],'additional_crease_projections_s':sum(samples[1:])}


def verify_acceleration(exe, out):
    """Check new CAD distances, the archived contact pose and signed-distance derivatives."""
    import trimesh
    from gear_field import FittedFaces, probe_set

    model = Path('results/aes/gear-field')
    source = Path('build/aes-nhrep')
    baseline = FittedFaces(model/'guide.npz',model/'faces.npz',compiled=False)
    accelerated = FittedFaces(model/'guide.npz',model/'faces.npz')
    if accelerated.refine is None:
        raise RuntimeError('Install Numba to benchmark the compiled projection')
    mesh = trimesh.load_mesh(source/'spur.obj',process=False)
    with np.load(source/'spur-geometry.npz') as geometry:
        probes = probe_set(mesh,geometry,5119)
    inputs, outputs = out/'probe-5119.txt', out/'truth-5119.csv'
    np.savetxt(inputs,probes,fmt='%.17g')
    subprocess.run([exe,'cad-point-distance','results/gears/models/spur-1a.step',inputs,outputs],check=True)
    cad = np.genfromtxt(outputs,delimiter=',',skip_header=1)
    with np.load('build/aes-nhrep-parametric/spur-reference.npz') as ref:
        points = np.vstack((probes,ref['points_mm']))
        reference = baseline.predict(points)
        measured = accelerated.predict(points)
        all_delta = 1000*np.max(abs(reference-measured))
        distances = measured[:len(probes)]
        archived = measured[len(probes):]
        loads = {}
        for pair in sorted(set(ref['contact_pair'])-{''}):
            ids = ref['contact_pair'] == pair
            weights = ref['weights_m'][ids]*1e6
            loads[pair] = {name:float(weights @ np.maximum(.055-d[ids],0))
                           for name,d in [('compiled_N',archived),('cad_N',ref['distance_mm'])]}
        archived_accuracy = accuracy(archived,ref['distance_mm'])
    ids = np.arange(0,len(probes),12)
    ids = ids[np.isfinite(cad[ids,1:]).all(axis=1)]
    h = 1e-5
    shifts = np.r_[np.eye(3),-np.eye(3)]*h
    v = accelerated.predict((probes[ids]+shifts[:,None,:]).reshape(-1,3)).reshape(6,len(ids))
    grad = ((v[:3]-v[3:])/(2*h)).T
    angle = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',grad,cad[ids,1:])/np.linalg.norm(grad,axis=1),-1,1)))
    fresh = accuracy(distances,cad[:,0])
    sign_errors = int(sum(np.sign(distances)!=np.sign(cad[:,0])))
    if fresh['within_0p125_um'] != len(probes) or sign_errors or all_delta > 1e-5:
        raise RuntimeError('Compiled projection differs from the reference or misses the CAD tolerance')
    return {'seed':5119,'fresh_points':len(probes),'fresh':fresh,'fresh_wrong_sign':sign_errors,
            'archived':archived_accuracy,'compiled_vs_numpy_max_um':float(all_delta),'loads':loads,
            'normal_checks':len(ids),'normal_p95_deg':float(np.percentile(angle,95)),'normal_max_deg':float(max(angle))}


def run_cad(exe, out, points, truth, indices, shuffle, pose):
    # One process retains the solid, face bounds and classifier across every query.
    warm = points[indices[:8]]
    blocks = [warm]
    for _ in range(3):
        blocks.extend((points[indices],points[shuffle],pose))
    query_file, result_file, times_file = (out/n for n in ('queries.txt','distances.csv','cad-times.csv'))
    np.savetxt(query_file,np.vstack(blocks),fmt='%.17g')
    subprocess.run([exe,'cad-point-distance','results/gears/models/spur-1a.step',query_file,result_file,times_file],
                   check=True)
    stats = np.loadtxt(times_file,delimiter=',',skiprows=1)
    dist = np.genfromtxt(result_file,delimiter=',',skip_header=1)[:,0]
    assert len(stats)==len(dist)==sum(map(len,blocks))
    single, batch, four = [], [], []
    offset = len(warm)
    for _ in range(3):
        end = offset+len(indices)
        single.append(stats[offset:end,1])
        offset = end
        end = offset+len(points)
        batch.append(float(stats[offset:end,1].sum()))
        values = dist[offset:end]
        np.testing.assert_allclose(values, truth[shuffle],rtol=0,atol=1e-9)
        offset = end
        end = offset+len(pose)
        four.append(float(stats[offset:end,1].sum()))
        offset = end
    return {**summary(np.array(single),batch,four,indices), **accuracy(values,truth[shuffle]),
            'setup_s':float(stats[0,0])}


def run_python(factory, points, truth, indices, shuffle, pose):
    start = perf_counter()
    predict = factory()
    init = perf_counter()-start
    start = perf_counter()
    predict(points[indices[:1]])
    first = perf_counter()-start
    predict(points[indices[:8]])
    single, batch, four = [], [], []
    for _ in range(3):
        times = []
        for i in indices:
            probe = points[i:i+1]
            start = perf_counter()
            predict(probe)
            times.append(perf_counter()-start)
        single.append(times)
        probes = points[shuffle]
        start = perf_counter()
        values = predict(probes)
        batch.append(perf_counter()-start)
        # Multiple short calls reduce clock noise, using the real four-point pose.
        reps = 8
        start = perf_counter()
        for _ in range(reps):
            predict(pose)
        four.append((perf_counter()-start)/reps)
    return {**summary(np.array(single),batch,four,indices), **accuracy(values,truth[shuffle]),
            'setup_s':init, 'first_query_s':first}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import torch
    import open3d as o3d
    import numba
    from gear_field import FittedFaces

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    folder = Path('results/aes/gear-field')
    sample_dir = Path('build/aes-gear-field')
    points = np.loadtxt(sample_dir/'probe-1407.txt')
    truth = np.genfromtxt(sample_dir/'truth-1407.csv',delimiter=',',skip_header=1)[:,0]
    assert points.shape == (1536,3) and truth.shape == (1536,)
    trial = Path('build/aes-nhrep-parametric')
    with np.load(trial/'spur-reference.npz') as archive:
        pose = archive['points_mm'][archive['group']=='contact']
    assert pose.shape == (4,3)
    rng = np.random.default_rng(825)
    indices = rng.permutation(np.r_[rng.choice(768,48,replace=False),768+rng.choice(768,48,replace=False)])
    shuffle = rng.permutation(len(points))

    def fitted():
        return FittedFaces(folder/'guide.npz',folder/'faces.npz').predict

    def numpy_fitted():
        return FittedFaces(folder/'guide.npz',folder/'faces.npz',compiled=False).predict

    def mesh():
        with np.load(folder/'guide.npz') as guide:
            vertices = o3d.core.Tensor(guide['vertices'].astype(np.float32))
            triangles = o3d.core.Tensor(guide['triangles'].astype(np.uint32))
        scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        scene.add_triangles(vertices,triangles)
        def predict(x):
            xyz = o3d.core.Tensor(x.astype(np.float32))
            return scene.compute_signed_distance(xyz,nthreads=1,nsamples=3).numpy()
        return predict

    def neural():
        field = torch.jit.load(str(trial/'gear_parametric_trial_spur_50k_model_h.pt'),map_location='cpu').eval()
        meta = json.loads((trial/'spur.json').read_text())
        scale, center = meta['mm_per_network_unit'], np.array(meta['center_mm'])
        def predict(x):
            with torch.inference_mode():
                v = torch.from_numpy(((x-center)/scale).astype(np.float32))
                return field(v).numpy().reshape(-1)*scale
        return predict

    report = {'seed':825, 'threads':1, 'repeats':3, 'single_points_per_repeat':len(indices),
              'points_per_batch':len(points),'pose_points':len(pose),'open3d_version':o3d.__version__,
              'torch_version':torch.__version__, 'numba_version':numba.__version__,
              'processor':os.environ.get('PROCESSOR_IDENTIFIER'),'methods':{}}
    with threadpool_limits(1):
        for name, factory in [('numpy_fitted_faces',numpy_fitted),('fitted_faces',fitted),
                              ('triangle_bvh_open3d',mesh),('nhrep',neural)]:
            print('Measuring',name,flush=True)
            report['methods'][name] = run_python(factory,points,truth,indices,shuffle,pose)
            print(json.dumps(report['methods'][name]),flush=True)
        print('Measuring native OCCT',flush=True)
        report['methods']['occt'] = run_cad(args.executable.resolve(),args.output,points,truth,indices,shuffle,pose)
        field = FittedFaces(folder/'guide.npz',folder/'faces.npz',compiled=False)
        report['numpy_projection_profile'] = profile_projections(field,points[shuffle])
        report['acceleration_validation'] = verify_acceleration(args.executable.resolve(),args.output)
    print(json.dumps(report['methods']['occt']),flush=True)
    (args.output/'timing.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
