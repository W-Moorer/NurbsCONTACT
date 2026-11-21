"""Time the same held-out decks and check prepared planar trim against direct OCCT rays."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from run import DT, MESH, read_csv, rms, run


def compare_geometry(root, direct):
    traces = []
    for file in sorted((root/"curves").glob("*.csv")):
        before = read_csv(direct/"curves"/file.name)
        current = read_csv(file)
        if not np.array_equal(before["measured_m"],current["measured_m"]):
            raise ValueError(f"observation changed for {file.name}")
        traces.append(float(np.max(np.abs(before["brep_m"]-current["brep_m"]))))
    first = read_csv(direct/"beam-comparison.csv")
    second = read_csv(root/"beam"/"comparison.csv")
    if not np.array_equal(first["measured_s"],second["measured_s"]):
        raise ValueError("steel beam digitization changed")
    result = {"trajectory_count":len(traces),"trajectory_max_abs_m":max(traces),
              "beam_duration_delta_us_rms":1e6*rms(first["brep_s"]-second["brep_s"])}
    if result["trajectory_max_abs_m"]>1e-7 or result["beam_duration_delta_us_rms"]>.01:
        raise RuntimeError(f"prepared planar trim changed the validated observations: {result}")
    return result


def measure(exe, root, direct):
    summary = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    durations = {item["key"]:item["duration_s"] for item in summary["trials"]}
    methods = [("brep-prepared",exe,0.),("triangle-bvh",exe,MESH)]
    result = {}
    if direct:
        methods += [("brep-direct",direct/"nc_contact.exe",0.)]
        prior=direct/"equivalence.json"
        if prior.is_file():
            result["planar_preparation_equivalence_at_development_steps"]=json.loads(prior.read_text())
    result["trials"]={}
    with TemporaryDirectory(prefix="nc-contact-timing-") as scratch:
        base=Path(scratch)/"run"
        settings=summary["settings"]
        configurations=[(x,root/"scenes"/(x+".scene"),durations[x],settings[key+"_dt_s"])
                        for x,key in (("bounce-06","bounce"),("plastic-06","slider"))]
        configurations += [("beam-27",root/"beam"/"scenes"/"impact-27.scene",.0005,1e-6)]
        for name,config,duration,dt in configurations:
            cases={method:[] for method,_,_ in methods}
            checked={}
            for i in range(7):
                order=methods[i%len(methods):]+methods[:i%len(methods)]
                for method, binary, deflection in order:
                    stride=1 if name.startswith("beam") else max(1,round(DT/dt))
                    trace,work=run(binary.resolve(),config,base,duration,dt,deflection,stride)
                    cases[method].append(work)
                    if not i:
                        checked[method]=(trace["q0"],trace["b1_z_m"],trace["p0_gap_m"])
            record={method:{"wall_median_s":float(np.median([x["wall_s"] for x in logs])),
                            "setup_median_s":float(np.median([x["setup_s"] for x in logs])),
                            "runs":logs} for method,logs in cases.items()}
            reference=record["brep-prepared"]["wall_median_s"]
            record["mesh_to_brep_ratio"]=record["triangle-bvh"]["wall_median_s"]/reference
            if "brep-direct" in record:
                record["direct_to_prepared_ratio"]=record["brep-direct"]["wall_median_s"]/reference
                delta=np.nanmax(np.abs(np.vstack(checked["brep-direct"])-np.vstack(checked["brep-prepared"])))
                if delta>1e-8:
                    raise RuntimeError(f"prepared trim changed a timed trajectory by {delta:.3g} m")
                record["direct_to_prepared_max_delta_m"]=float(delta)
            record["integration_step_s"]=dt
            result["trials"][name]=record
    (root/"timing.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    for name,data in result["trials"].items():
        print(name,{key:value for key,value in data.items() if key.endswith("ratio")})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe",type=Path)
    parser.add_argument("results",type=Path)
    parser.add_argument("--direct",type=Path)
    args=parser.parse_args()
    measure(args.exe,args.results,args.direct)
