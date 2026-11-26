"""Count OCCT fallback reasons on the same 800 archived poses per reducer stage."""
import argparse
import json
from pathlib import Path
import subprocess

from cpu_pressure_columns import digest, fixed_geometry_inputs, stage_poses


REASONS=("no_fast_query","no_compiled_spans","near_singular_projection","contraction_bound",
         "initial_uv_outside_span","invalid_weight","root_interval_touches_edge","multiple_roots",
         "root_near_segment_end","degenerate_normal","singular_newton_step","newton_leaves_span",
         "iteration_limit","nonrectangular_trim","near_trim_or_seam")


def replay(scene,archive,exe,output,relocate=False):
    trajectory,indices,_,bodies,pairs,angles,origins=fixed_geometry_inputs(scene,archive)
    reference=json.loads((archive/"depth-bounds/poses-selected.json").read_text())["stages"]
    stages=[]
    for stage,(a,b,_,limit) in enumerate(pairs):
        poses=stage_poses(bodies,angles,origins,indices,a,b)
        payload="".join(" ".join(format(x,".17g") for x in pose)+"\n" for pose in poses)
        cmd=[exe,bodies[a][0],bodies[b][0],"6",str(2*limit),"--loaded","--bezier","--depth-bounds"]
        if relocate:
            cmd.append("--relocate")
        run=subprocess.run(cmd,input=payload,capture_output=True,text=True,check=True,timeout=1800)
        rows=[json.loads(line) for line in run.stdout.splitlines()]
        reasons=[]
        cohort=[]
        for line in run.stderr.splitlines():
            if line.startswith("uv_cohort "):
                site,calls,resolved,discordant,direct_ns,native_ns=map(int,line.split()[1:])
                cohort.append(dict(site=("first_exit","interior_segment")[site],calls=calls,resolved=resolved,
                                   discordant=discordant,local_s=direct_ns*1e-9,
                                   avoided_native_s=native_ns*1e-9))
            if line.startswith("native_fallback "):
                site,cause,count,finished,hits,ns=map(int,line.split()[1:])
                reasons.append(dict(site=("first_exit","interior_segment")[site],reason=REASONS[cause],
                                    calls=count,finished=finished,with_intersection=hits,perform_s=ns*1e-9))
        expected=reference[stage]["current"]["ray_counts"][2]
        if len(rows)!=len(indices) or not all(row["resolved"] for row in rows):
            raise ValueError("Incomplete fixed-pose replay")
        native_calls=rows[-1]["ray_counts"][2]
        if (not relocate and native_calls!=expected) or sum(r["calls"] for r in reasons)!=native_calls:
            raise ValueError("Native calls differ from control")
        if cohort and not relocate and sum(r["calls"] for r in cohort)!=sum(
                r["calls"] for r in reasons if r["reason"]=="initial_uv_outside_span"):
            raise ValueError("Paired rays differ from the original UV fallback cohort")
        stages.append(dict(stage=stage+1,poses=len(rows),native_calls=native_calls,reasons=reasons,
                           same_ray_cohort=cohort))
    output.write_text(json.dumps(dict(scene_sha256=digest(scene),trajectory_sha256=digest(trajectory),
                                     executable_sha256=digest(exe),stages=stages),indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("scene","archive","exe","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--relocate",action="store_true")
    replay(**vars(parser.parse_args()))
