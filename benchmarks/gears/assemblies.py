"""Export named STEP assemblies at the measured snapshot poses."""

import argparse
import os
from pathlib import Path

import cadquery as cq
import numpy as np

from analyze import track
from bearing_run import CAGE_RPM, ROLLERS, track as bearing_track
from run import CASES, csv_rows


def shape(path):
    solid=cq.importers.importStep(str(path)).val()
    if len(solid.Solids())!=1 or not solid.isValid():
        raise ValueError(f"Invalid imported STEP solid {path}")
    return solid


def add(assembly,solid,name,color,xyz,angle=0.):
    location=cq.Location(cq.Vector(*xyz),cq.Vector(0,0,1),float(angle))
    assembly.add(solid,name=name,loc=location,color=cq.Color(*color))


def export(assembly,path,count):
    assembly.export(str(path),exportType="STEP",mode="default")
    loaded=cq.importers.importStep(str(path)).val()
    if len(loaded.Solids())!=count:
        raise ValueError(f"Unexpected occurrence count after STEP round trip for {path}")
    print(f"Exported {path} with {count} solid occurrences",flush=True)


def gears(root,folder):
    for case in CASES:
        layout=csv_rows(root/"models"/f"{case}.csv")
        trace=track(root,case+"-cad")
        row=trace[np.searchsorted(trace["time"],.133)]
        assembly=cq.Assembly(name=case)
        for gear in layout:
            shaft=int(gear["shaft"])
            angle=float(gear["angle_zero_rad"])+float(gear["direction"])*(
                float(gear["start_rpm"])*2*np.pi/60*row["time"]+row[f"q{shaft}"]
            )
            xyz=(float(gear["center_x_mm"]),0.,float(gear["z_mm"]))
            color=(.22,.47,.69) if gear["side"]=="a" else (.55,.65,.73)
            name=f"stage-{gear['stage']}-shaft-{shaft}-{gear['side']}"
            add(assembly,shape(root/"models"/gear["step"]),name,color,xyz,np.degrees(angle))
        export(assembly,folder/f"{case}.step",len(layout))


def bearing(root,folder):
    data=root/"bearing"
    trace=bearing_track(data,"cad")
    row=trace[np.searchsorted(trace["time"],.190)]
    assembly=cq.Assembly(name="cylindrical-roller-bearing")
    models=data/"models"
    add(assembly,shape(models/"outer.step"),"outer-race",(.52,.62,.70),(0.,0.,0.))
    add(assembly,shape(models/"inner.step"),"inner-race",(.19,.43,.65),
        (row["x_um"]/1000,row["y_um"]/1000,0.))
    roller=shape(models/"roller.step")
    for i in range(ROLLERS):
        theta=2*np.pi*(CAGE_RPM*row["time"]/60+i/ROLLERS)
        radius=row[f"radius{i}_mm"]
        add(assembly,roller,f"roller-{i+1:02d}",(.55,.72,.77),
            (radius*np.cos(theta),radius*np.sin(theta),0.))
    export(assembly,folder/"bearing.step",ROLLERS+2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    root=parser.parse_args().results
    folder=root/"assemblies"
    folder.mkdir(exist_ok=True)
    gears(root,folder)
    bearing(root,folder)


if __name__=="__main__":
    main()
    if os.name=="nt":
        # All STEP files have been closed and round-tripped at this point.
        os._exit(0)
