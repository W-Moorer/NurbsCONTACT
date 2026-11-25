"""Replay native STEP witnesses and optimized mesh regions at identical CPU poses."""
import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from _gpu_source import read_model
from cpu_backend_run import digest


def load_torque(regions, k, center, axis, limit):
    if regions is None:
        return np.nan,np.nan
    if not len(regions):
        return 0.,0.
    r=np.asarray(regions)
    if not np.isfinite(r).all() or np.any((r[:,6]<=0)|(r[:,6]>limit)):
        return np.nan,np.nan
    forces=r[:,3:6]*(k*r[:,6:7]/len(r))
    return k*r[:,6].mean(), np.cross(r[:,:3]-center,forces).sum(axis=0)@axis


def corrected(seeds, chords):
    if seeds is None or any(c["depth_m"] is None for c in chords):
        return None
    out=[]
    for seed,c in zip(seeds,chords,strict=True):
        a,b=c["offsets_m"]
        n=np.asarray(seed[3:6])
        out.append([*(np.asarray(seed[:3])+.5*(a-b)*n),*n,c["depth_m"]])
    return out


def representatives(witnesses,labels):
    # A diagnostic for sensitivity to the choice of application point.
    # Pick a measured point from each joined group, including when groups split.
    selected={}
    for r,label in zip(witnesses,labels,strict=True):
        if label<0:
            return None
        if label not in selected or r[6]>selected[label][6]:
            selected[label]=r
    return list(selected.values())


def records(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def group_input(pose, regions, reach):
    return " ".join([pose.rstrip(), str(reach), str(len(regions)),
                     *(format(x, ".17g") for r in regions for x in r[:6])]) + "\n"


def invoke(args, command, source, dest):
    start = time.perf_counter()
    with source.open() as inp, dest.open("w") as log, dest.with_suffix(dest.suffix + ".err").open("w") as err:
        subprocess.run(command, stdin=inp, stdout=log, stderr=err, timeout=args.budget, check=True)
    return time.perf_counter() - start


def geometry(args, stage, bodies, pair):
    a, b, _, limit = pair
    assets = [bodies[j][0] for j in (a, b)]
    prefix = args.work / str(stage)
    file = lambda suffix: prefix.with_suffix(suffix)
    with np.load(args.trajectory) as inputs:
        origin, angle = inputs["origins_m"], inputs["angles_rad"]
        columns = [x for j in (a, b) for x in
                   [*origin[:, j, :].T, *[np.full(len(angle), v) for v in bodies[j][2]], angle[:, j]]]
    np.savetxt(file(".poses"), np.column_stack(columns), fmt="%.17g")
    times = {}
    times["mesh"] = invoke(args, [args.mesh, "--method", "mesh-obb", "--regions", "contour-ray", "--axis-angle",
                                "--keep-unresolved", "--max-depth-m", str(limit), *[p.with_suffix(".msh") for p in assets]],
                          file(".poses"), file(".jsonl"))
    with file(".poses").open() as poses, file(".seeds").open("w") as dest:
        for pose, row in zip(poses, records(file(".jsonl")), strict=True):
            dest.write(group_input(pose, row["regions"] or [], 2 * limit))
    times["local"] = invoke(args, [args.cad, *assets, "--local-chords"], file(".seeds"), file(".local"))
    times["native_including_group"] = invoke(args, [args.probes, *assets, "6", str(2 * limit), "--groups"],
                                            file(".poses"), file(".integrated"))
    file(".times").write_text(json.dumps(times))
    file(".binaries").write_text(json.dumps({exe.name:digest(exe) for exe in (args.mesh,args.cad,args.probes)}))


def audit_indices(counts):
    n = len(counts) - 1
    ids = {0, *range(1, min(11, n+1)), n, *(round(n*f) for f in (.2, .4, .5, .6, .8))}
    ids.update(j for j, count in enumerate(counts) if count == 0)
    # Include the peak already identified in the archived reducer trace.
    if n == 50000:
        ids.add(46315)
    for start in (20, (n+1)//3, 2*((n+1)//3)):
        k = next((i for i in range(max(1,start), n+1)
                  if min(counts[i-1], counts[i]) >= 1 and counts[i] != counts[i-1]), None)
        if k is not None:
            ids.update((k-1, k))
    cases = [j for j, count in enumerate(counts) if count < 0][:2]
    rare = next((j for j, count in enumerate(counts) if j > 300 and count >= 5), None)
    for j in cases + ([] if rare is None else [rare]):
        ids.update(i for i in (j-1, j, j+1) if 0 <= i <= n)
    return sorted(ids)


def pick(rows, ids):
    return {i: row for i, row in enumerate(rows) if i in ids}


def audit(args, stage, bodies, pair):
    a, b, _, limit = pair
    assets = [bodies[j][0] for j in (a,b)]
    file = lambda suffix: (args.work / str(stage)).with_suffix(suffix)
    counts = [-1 if r["regions"] is None else len(r["regions"]) for r in records(file(".jsonl"))]
    ids = audit_indices(counts)
    selected = set(ids)
    mesh = pick(records(file(".jsonl")), selected)
    native = pick(records(file(".integrated")), selected)
    poses = file(".poses").read_text().splitlines()
    data = {}
    for j in ids:
        data[j] = dict(mesh=mesh[j]["regions"], native=[s[2:] for s in native[j]["witnesses"]])
    membership_input = file(".membership-in")
    with membership_input.open("w") as out:
        for j, row in data.items():
            seeds = (row["mesh"] or []) + row["native"]
            print(poses[j], len(seeds), *(x for r in seeds for x in r[:3]), file=out)
    times = {"membership": invoke(args, [args.cad, *assets, "--memberships"], membership_input, file(".membership"))}
    for j, ref in zip(ids, records(file(".membership")), strict=True):
        row = data[j]
        row["reference"] = ref
        if ref["resolved"]:
            split = len(row["mesh"] or [])
            row["mesh_labels"] = ref["labels"][:split]
            row["native_labels"] = ref["labels"][split:]
    # Run the registered method's region factory independently to obtain force
    # arms and depths. A failed CAD query stays unresolved.
    file(".audit-poses").write_text("".join(poses[j]+"\n" for j in ids))
    times["local_reference"] = invoke(args, [args.cad, *assets, "--profile-local-regions", str(limit)],
                                      file(".audit-poses"), file(".native-ref"))
    for j, result in zip(ids, records(file(".native-ref")), strict=True):
        data[j]["cad_regions"] = result
    # Keep the mesh connectivity diagnostic for the same audited poses.
    file(".mesh-groups-in").write_text("".join(group_input(poses[j],data[j]["mesh"] or [],2*limit) for j in ids))
    invoke(args, [args.cad, *assets, "--group-chords"], file(".mesh-groups-in"), file(".mesh-groups"))
    for j, result in zip(ids, records(file(".mesh-groups")), strict=True):
        data[j]["mesh_join"] = result
    file(".audit.json").write_text(json.dumps(dict(rows=data,process_s=times), allow_nan=False))


def fixed(args, stage, bodies, pair):
    a,b,_,limit = pair
    paths = [bodies[j][0] for j in (a,b)]
    profile = json.loads(args.profile.read_text())
    info = profile["stages"][stage]
    for path in paths:
        if digest(path) != info["assets"][path.name]:
            raise ValueError("STEP input differs from fixed-pose reference")
    file = lambda suffix: (args.work / str(stage)).with_suffix(suffix)
    poses = file(".poses").read_text().splitlines()
    measured = []
    for name, run in info["runs"].items():
        rows = []
        occupied = set()
        for row in run["rows"]:
            if row["step"] not in occupied:
                rows.append(row)
                occupied.add(row["step"])
        file(".fixed-groups").write_text("".join(group_input(poses[r["step"]],r["regions"],2*limit) for r in rows))
        with file(".fixed-rays").open("w") as out:
            for r in rows:
                for region in r["regions"]:
                    print(poses[r["step"]],*region[:6],2*limit,file=out)
        invoke(args,[args.cad,*paths,"--local-chords"],file(".fixed-groups"),file(".fixed-local"))
        invoke(args,[args.cad,*paths,"--normal-chords"],file(".fixed-rays"),file(".fixed-whole"))
        full = list(records(file(".fixed-whole")))
        local = list(records(file(".fixed-local")))
        flat = [(r["step"],k,c,old) for r,cs in zip(rows,local,strict=True)
                for k,(c,old) in enumerate(zip(cs,r["cad_normal_chords"],strict=True))]
        for (j,k,c,old),ref in zip(flat,full,strict=True):
            measured.append(dict(seed=name,step=j,region=k,local=c,whole=ref,archived_m=old["depth_m"]))
    file(".fixed.json").write_text(json.dumps(measured,allow_nan=False))


def swap(args,stage,bodies,pair):
    a,b,stiffness,limit=pair
    file=lambda suffix:(args.work/str(stage)).with_suffix(suffix)
    mesh_counts=[-1 if r["regions"] is None else len(r["regions"]) for r in records(file(".jsonl"))]
    ids=audit_indices(mesh_counts)
    poses=file(".poses").read_text().splitlines()
    result={}
    for reverse,name in [(False,"forward"),(True,"reverse")]:
        path=file(".swap-in")
        with path.open("w") as out:
            for j in ids:
                fields=poses[j].split()
                print(" ".join(fields[7:]+fields[:7] if reverse else fields),file=out)
        source,target=(b,a) if reverse else (a,b)
        log=file(".swap-"+name)
        invoke(args,[args.probes,bodies[source][0],bodies[target][0],"6",str(2*limit),"--groups"],path,log)
        values=[]
        for j,row in zip(ids,records(log),strict=True):
            w=[s[2:] for s in row["witnesses"]]
            if reverse:
                w=[[*p[:3],*[-x for x in p[3:6]],p[6]] for p in w]
            r=representatives(w,row["labels"])
            center=np.asarray(poses[j].split()[7:10],dtype=float)
            load,torque=load_torque(r,stiffness,center,bodies[b][2],limit)
            values.append(dict(step=j,groups=len(r),load_N=float(load) if np.isfinite(load) else None,
                               torque_Nm=float(torque) if np.isfinite(torque) else None))
        result[name]=values
    file(".swap.json").write_text(json.dumps(result,allow_nan=False))


def normals(args,stage,bodies,pair):
    a,b,k,limit=pair
    cmd=[args.cad,bodies[a][0],bodies[b][0]]
    file=lambda suffix:(args.work/str(stage)).with_suffix(suffix)
    poses=file(".poses").read_text().splitlines()
    audit_rows=json.loads(file(".audit.json").read_text())["rows"]
    ids=sorted(map(int,audit_rows))
    seeds=[audit_rows[str(j)]["native"] for j in ids]
    file(".directions-in").write_text("".join(group_input(poses[j],rs,2*limit) for j,rs in zip(ids,seeds,strict=True)))
    invoke(args,[*cmd,"--group-chords"],file(".directions-in"),file(".directions-groups"))
    selected=[]
    for j,rs,group in zip(ids,seeds,records(file(".directions-groups")),strict=True):
        best={}
        for r,c,label in zip(rs,group["chords"],group["labels"],strict=True):
            if label<0:
                continue
            score=(c["opposition"],c["depth_m"])
            if label not in best or score>best[label][0]:
                best[label]=(score,r,c)
        region=[r for _,r,_ in best.values()]
        chords=[c for _,_,c in best.values()]
        revised=corrected(region,chords)
        bisectors=[[*r[:3],*c["bisector"]] for r,c in zip(revised,chords,strict=True)]
        selected.append((revised,bisectors,[c["faces"] for c in chords],-1 not in group["labels"]))
    file(".bisectors-in").write_text("".join(group_input(poses[j],s[1],2*limit) for j,s in zip(ids,selected,strict=True)))
    invoke(args,[*cmd,"--local-chords"],file(".bisectors-in"),file(".directions-recast"))
    results=[]
    for j,(revised,lines,faces,valid),hits in zip(ids,selected,records(file(".directions-recast")),strict=True):
        center=np.asarray(poses[j].split()[7:10],dtype=float)
        def response(regions):
            values=load_torque(regions,k,center,bodies[b][2],limit)
            return list(values) if np.isfinite(values).all() else None
        same=all(c.get("faces")==face for c,face in zip(hits,faces,strict=True))
        results.append(dict(step=j,opposed_source=response(revised) if valid else None,
            opposed_bisector=response(corrected(lines,hits)) if valid and same else None))
    file(".normals.json").write_text(json.dumps(results,allow_nan=False))


def failures(args,stage,bodies,pair):
    a,b,_,limit=pair
    file=lambda suffix:(args.work/str(stage)).with_suffix(suffix)
    ids=[]
    with file(".failed-rays").open("w") as out,file(".poses").open() as poses:
        for j,(pose,m,cs) in enumerate(zip(poses,records(file(".jsonl")),records(file(".local")),strict=True)):
            for k,(seed,chord) in enumerate(zip(m["regions"] or [],cs,strict=True)):
                if chord["depth_m"] is None:
                    ids.append((j,k))
                    print(pose.rstrip(),*seed[:6],2*limit,file=out)
    wall=invoke(args,[args.cad,bodies[a][0],bodies[b][0],"--normal-chords"],file(".failed-rays"),file(".failed-ref"))
    counts={}
    resolved=[]
    for identity,r in zip(ids,records(file(".failed-ref")),strict=True):
        counts[r["status"]]=counts.get(r["status"],0)+1
        if r["depth_m"] is not None:
            resolved.append(dict(step=identity[0],region=identity[1],depth_m=r["depth_m"]))
    file(".failed.json").write_text(json.dumps(dict(examined=len(ids),whole_step_status=counts,
        recovered=resolved,process_s=wall),allow_nan=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["geometry","fixed","audit","swap","normals","failures"])
    for key in ("scene","trajectory","work","mesh","cad","probes"):
        parser.add_argument(key,type=Path)
    parser.add_argument("--profile",type=Path,default=Path("results/aes/cpu-cad-reducer-profile/profile.json"))
    parser.add_argument("--budget",type=float,default=1800.)
    args=parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    bodies,pairs=read_model(args.scene)
    operation={"geometry":geometry,"fixed":fixed,"audit":audit,"swap":swap,"normals":normals,"failures":failures}[args.action]
    with ThreadPoolExecutor(min(2,len(pairs))) as pool:
        list(pool.map(lambda item: operation(args,item[0],bodies,item[1]),enumerate(pairs)))


if __name__=="__main__":
    main()
