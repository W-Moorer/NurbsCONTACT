"""Render the measured-profile STEP gears and the revised NASA HHT trajectory."""

import argparse
import json
from pathlib import Path
import sys

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render import Assembly, SIZE, annotation, reserve_caption

from compare import NASA, read


def render(root, case):
    trace = read(root/"runs"/f"nasa-{case}-brep.csv")
    start = trace["time"][0] + .75*(trace["time"][-1]-trace["time"][0])
    end = start+4*60/(NASA[case][0]*28)
    rows = trace[np.searchsorted(trace["time"], np.linspace(start, end, 96))]
    folder = root/"animations"
    folder.mkdir(exist_ok=True)
    thumbnails = root/"images"
    thumbnails.mkdir(exist_ok=True)
    model = "rig28-flex"
    title = f"NASA 28/28 gear pair · {NASA[case][0]:,} rpm"
    base = .003175*14*np.cos(np.radians(20))
    pitch_projection_mm = 44.45*np.sin(np.radians(20))
    outputs = []
    for view in ("isometric",):
        assembly = Assembly(root.parent, model, view)
        path = folder/f"nasa-{case}-{view}.mp4"
        try:
            assembly.plotter.show(auto_close=False, interactive=False)
            reserve_caption(assembly.plotter)
            with av.open(str(path), "w", options={"movflags":"faststart"}) as output:
                stream = output.add_stream("libx264", rate=24)
                stream.width, stream.height = SIZE
                stream.pix_fmt = "yuv420p"
                for i, row in enumerate(rows):
                    forces = np.nan_to_num([row[f"force{j}_N"] for j in range(4)])
                    strongest = int(np.argmax(forces))
                    location = pitch_projection_mm-base*1000*np.radians(row[f"roll{strongest}_deg"])
                    station = location if forces[strongest]>0 else np.nan
                    pinion = row["pinion_angle_rad"]
                    image = assembly.frame(0, [pinion, pinion+row["te_um"]*1e-6/base], [(station, 0.)])
                    pixels = annotation(image, title, view, row["time"], [sum(forces)], NASA[case][1], start, end)
                    if i==24:
                        image.save(thumbnails/f"nasa-{case}-{view}.png")
                    for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
        finally:
            assembly.close()
        outputs.append({"view":view,"path":path.name,"frames":len(rows),"fps":24,
                        "first_sample_s":float(rows["time"][0]),"last_sample_s":float(rows["time"][-1])})
    (folder/"index.json").write_text(json.dumps(outputs,indent=2)+"\n")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    parser.add_argument("--case",choices=NASA,default="b")
    args = parser.parse_args()
    render(args.results,args.case)
