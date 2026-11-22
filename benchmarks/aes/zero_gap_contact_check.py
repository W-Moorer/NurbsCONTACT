"""Audit first loading and release on a completed zero-gap gear trajectory."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np
from gear_contact_paths import angle
from gpu_mesh_line import read_full_mesh
from gpu_mesh_pairs import MeshPair
from nhrep_gear import read_occt_solid


def check(args):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopLoc import TopLoc_Location
    from OCP.gp import gp_Trsf

    with (args.models/'spur.csv').open(newline='') as stream:
        a, b = csv.DictReader(stream)
    with args.trace.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    loads = np.asarray([float(row['normal0_N']) for row in rows])
    entry = int(np.flatnonzero(loads>0)[0])
    release = entry + int(np.flatnonzero(loads[entry:]==0)[0])
    if entry == 0:
        raise ValueError('Expected an initially separated pair.')
    peak = entry + int(np.argmax(loads[entry:release]))
    samples = [('initial',0), ('before_loading',entry-1), ('first_loaded',entry),
               ('first_peak',peak), ('before_release',release-1), ('first_released',release)]
    pair = MeshPair(read_full_mesh(args.mesh_a), read_full_mesh(args.mesh_b))
    solid_a, solid_b = [read_occt_solid(args.models/part['step']) for part in (a,b)]
    dx = float(b['center_x_mm'])-float(a['center_x_mm'])
    dz = float(b['z_mm'])-float(a['z_mm'])
    records = []
    for label, i in samples:
        row = rows[i]
        aa, ab = angle(a,row), angle(b,row)
        c, s = np.cos(ab-aa), np.sin(ab-aa)
        T = np.array([[c,-s,0,dx*np.cos(aa)], [s,c,0,-dx*np.sin(aa)], [0,0,1,dz]])
        transform = gp_Trsf()
        transform.SetValues(*T.ravel())
        start = perf_counter()
        common = BRepAlgoAPI_Common(solid_a, solid_b.Moved(TopLoc_Location(transform)))
        if not common.IsDone() or common.Shape().IsNull():
            raise RuntimeError(f'STEP intersection failed on {label}')
        mass = GProp_GProps()
        BRepGProp.VolumeProperties_s(common.Shape(),mass,True)
        volume = mass.Mass()
        cad_s = perf_counter()-start
        if not np.isfinite(volume) or volume<0:
            raise ValueError(f'Invalid common volume on {label}')
        pose = cp.asarray(T)
        pairs = int(pair.query(pose,0.,1e-10,max(abs(dx),abs(dz)),True).get())
        loaded = bool(loads[i]>0)
        records.append(dict(frame=label,time_s=float(row['time']),pose_mm=T.tolist(),
                            signed_flank_gap_um=1000*float(row['mesh_gap0_mm']),
                            force_N=loads[i],common_volume_mm3=volume,mesh_intersecting_pairs=pairs,
                            cad_boolean_s=cad_s,agreement=loaded==(volume>0)==(pairs>0)))
    return dict(trace_sha256=sha256(args.trace.read_bytes()).hexdigest(),
                mesh_sha256=[sha256(p.read_bytes()).hexdigest() for p in (args.mesh_a,args.mesh_b)],
                gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),samples=records)


def replay(args):
    reference = json.loads(args.reference.read_text())
    if reference['mesh_sha256'] != [sha256(p.read_bytes()).hexdigest() for p in (args.mesh_a,args.mesh_b)]:
        raise ValueError('The replay requires the two audited meshes.')
    pair = MeshPair(read_full_mesh(args.mesh_a), read_full_mesh(args.mesh_b))
    poses = [(cp.asarray(row['pose_mm']), np.max(np.abs(np.asarray(row['pose_mm'])[:,3])))
             for row in reference['samples']]
    samples = [dict(frame=r['frame'],time_s=r['time_s'],intersections=r['mesh_intersecting_pairs'],
                    elapsed_ms=[]) for r in reference['samples']]
    begin,end = cp.cuda.Event(),cp.cuda.Event()
    for repeat in range(21):
        for row, (pose,scale) in zip(samples,poses,strict=True):
            begin.record()
            count = pair.query(pose,0.,1e-10,scale,True)
            end.record()
            end.synchronize()
            if int(count.get()) != row['intersections']:
                raise RuntimeError('Intersection count differs from the reference replay.')
            if repeat:
                row['elapsed_ms'].append(cp.cuda.get_elapsed_time(begin,end))
    return dict(gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),samples=samples,
                reference_sha256=sha256(args.reference.read_bytes()).hexdigest(),
                median_ms=float(np.median([t for r in samples for t in r['elapsed_ms']])))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    audit, timing = commands.add_parser('audit'),commands.add_parser('replay')
    for command, arguments in [(audit,('models','trace','mesh_a','mesh_b','output')),
                               (timing,('reference','mesh_a','mesh_b','output'))]:
        for name in arguments:
            command.add_argument(name,type=Path)
    args = parser.parse_args()
    result = check(args) if args.command=='audit' else replay(args)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n', encoding='utf-8')
    if args.command=='audit' and not all(row['agreement'] for row in result['samples']):
        raise SystemExit('Whole-solid checks disagree with one or more sampled gear loads.')
