"""List the four part-level cache inventories and test warm loading for each reducer stage."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from _gpu_source import read_model
from cpu_backend_run import digest


def audit(scene,probes,cache,depth_manifest,output):
    bodies,pairs=read_model(scene)
    assets=json.loads(depth_manifest.read_text())["tables"]
    kinds=("bezier-rays-v2","charts-6","sites-6","tiles-6","area12","column-depth-v1")
    sizes={kind:0 for kind in kinds}
    for row in assets:
        tab=cache/row["name"]
        if digest(tab)!=row["sha256"]:
            raise ValueError(f"Shape cache differs from archived depth table: {tab}")
        prefix=tab.name.removesuffix("column-depth-v1.bin")
        for kind in kinds:
            sizes[kind]+=(cache/(prefix+kind+".bin")).stat().st_size
    dense={}
    for order in (24,48,96):
        files=list(cache.glob(f"*-quadrature-{order}.bin"))
        dense[str(order)]=dict(files=len(files),bytes=sum(item.stat().st_size for item in files))
    env={**os.environ,"NC_CAD_CACHE_DIR":str(cache.resolve())}
    warm=[]
    for a,b,_,limit in pairs:
        cmd=[probes,bodies[a][0],bodies[b][0],"6",str(2*limit),"--prepare","--bezier","--depth-bounds"]
        result=subprocess.run(cmd,env=env,check=True,capture_output=True,text=True,timeout=1800)
        warm.append(dict(parts=[bodies[a][0].name,bodies[b][0].name],**json.loads(result.stdout)))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(scene_sha256=digest(scene),probes_sha256=digest(probes),
                         parts=len(assets),part_table_bytes=sizes,cache_directory_dense_tables=dense,
                         warm_load=warm),indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("scene","probes","cache","depth_manifest","output"):
        parser.add_argument(name,type=Path)
    audit(**vars(parser.parse_args()))
