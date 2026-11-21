"""Compare the common STEP solver to the electrical impact durations in Qi and Yin Fig. 17."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw

from run import MESH, ZERO, rms, run, values


PAPER = "https://journals.sagepub.com/doi/10.1177/1687814016668089"
FIGURE = ("https://journals.sagepub.com/cms/10.1177/1687814016668089/asset/"
          "4e69ce0f-e2f6-43de-a0d9-251d4bf2752e/assets/images/large/10.1177_1687814016668089-fig17.jpg")
# Centres of the experiment's black disks, measured in the 833 x 478 publication image.
PIXELS = [(201,218),(209,218),(216,212),(225,209),(234,215),(242,209),(251,206),
          (260,206),(270,205),(281,203),(292,202),(303,200),(315,197),(327,197),
          (339,190),(352,188),(367,182),(382,178),(398,176),(415,179),(433,171),
          (453,173),(474,162),(496,160),(521,158),(548,155),(577,143)]
R = .035
MASS = 1.4008
BEAM_MASS = 10.2132/2
MODE_HZ = 99.9
HERTZ = (4/3)*np.sqrt(R)/((1-.33**2)/208e9+(1-.30**2)/206e9)
DT = 1e-6


def measurements(image_file, root):
    if not image_file.is_file():
        image_file.parent.mkdir(parents=True,exist_ok=True)
        request = Request(FIGURE,headers={"User-Agent":"Mozilla/5.0"})
        with urlopen(request,timeout=30) as response:
            image_file.write_bytes(response.read())
    with Image.open(image_file) as source:
        if source.size!=(833,478):
            raise ValueError("the source image changed")
        proof = source.convert("RGB")
    draw = ImageDraw.Draw(proof)
    for x,y in PIXELS:
        draw.ellipse((x-2,y-2,x+2,y+2),fill="#e03b58")
    proof.save(root/"digitization-qa.png")
    x,y = np.asarray(PIXELS).T
    v = (.8+.2*(x-125)/(796-125))**-5
    duration = (120+120*(380-y)/(380-20))*1e-6
    np.savetxt(root/"measured.csv",np.column_stack((v,duration,x,y)),delimiter=",",comments="",
               header="impact_velocity_m_s,contact_duration_s,pixel_x,pixel_y",fmt="%.10g")
    return v,duration


def deck(velocity, geometry):
    k = BEAM_MASS*(2*np.pi*MODE_HZ)**2
    return "\n".join((
        "nc-contact-v1", "gravity 0 0 -9.81 2 2 1",
        values("coordinate",BEAM_MASS,0,0,k,0,0,0),
        values("coordinate",MASS,R+4e-6,-velocity,0,0,0,0),
        # Body 0 mass is zero for gravity because its coordinate is a perturbation about static equilibrium.
        values("body",json.dumps((geometry/"steel-beam.step").as_posix()),
               0,0,0,0,0,0,1,0,0,0,0,-1,0,0,0,1,0,0,0,1),
        values("body",json.dumps((geometry/"steel-ball.step").as_posix()),
               MASS,0,0,0,0,0,1,0,0,0,0,-1,0,0,0,1,1,0,0,1),
        values("probe",0,1,"world",*ZERO,*ZERO,0,0,1,HERTZ,1.5,0,0,.0005,.003),
    ))+"\n"


def event_duration(trace):
    g = trace["p0_gap_m"]
    t = trace["time_s"]
    begins = np.flatnonzero((g[:-1]>=0)&(g[1:]<0))
    ends = np.flatnonzero((g[:-1]<0)&(g[1:]>=0))
    if not len(begins) or not len(ends) or ends[0]<=begins[0]:
        raise ValueError("first complete duration did not resolve")

    def crossing(i):
        return float(t[i]-(t[i+1]-t[i])*g[i]/(g[i+1]-g[i]))

    return crossing(ends[0])-crossing(begins[0])


def compare(exe, root, image_file):
    models = root/"models"
    output = root/"beam"
    for folder in (models, output, output/"runs", output/"scenes"):
        folder.mkdir(parents=True,exist_ok=True)
    for name, shape in (("steel-ball",("sphere",1000*R)),("steel-beam",("box",780,60,27.8,-13.9))):
        subprocess.run([str(exe),"make",*map(str,shape),str(models/f"{name}.step")],
                       capture_output=True,text=True,check=True)
    speeds, measured = measurements(image_file,output)
    branches = (("brep",0.,DT),("mesh",MESH,DT),("brep_half_step",0.,DT/2))
    predicted = {name:[] for name,_,_ in branches}
    details = []
    for i,v in enumerate(speeds,1):
        scene = output/"scenes"/f"impact-{i:02}.scene"
        scene.write_text(deck(v,Path("../../models")),encoding="utf-8")
        record = {"id":i,"impact_velocity_m_s":float(v)}
        for name, mesh, dt in branches:
            trace,work = run(exe,scene,output/"runs"/f"impact-{i:02}-{name}",.00050,dt,mesh)
            duration = event_duration(trace)
            predicted[name].append(duration)
            record[name] = work
        details.append(record)
    data = np.column_stack((speeds,measured,*(predicted[name] for name,_,_ in branches)))
    np.savetxt(output/"comparison.csv",data,delimiter=",",comments="",fmt="%.12g",
               header="impact_velocity_m_s,measured_s,brep_s,mesh_s,brep_half_step_s")
    errors = {}
    for column,(name,_,_) in enumerate(branches,2):
        delta = data[:,column]-measured
        errors[name] = {"rmse_us":1e6*rms(delta),"mean_absolute_error_us":1e6*float(np.mean(np.abs(delta))),
                        "maximum_absolute_error_us":1e6*float(np.max(np.abs(delta)))}
    result = {"paper":PAPER,"source_figure":FIGURE,"sha256":hashlib.sha256(image_file.read_bytes()).hexdigest(),
              "observation":"electrical contact duration of first pulse at beam centre",
              "trials":len(speeds),"Hertz_stiffness_N_m_power_1p5":HERTZ,"training_trials":0,
              "beam_effective_mass_kg":BEAM_MASS,"beam_mode_Hz":MODE_HZ,
              "duration_errors":errors,"mesh_to_brep_us_rms":1e6*rms(data[:,3]-data[:,2]),
              "half_step_to_brep_us_rms":1e6*rms(data[:,4]-data[:,2]),"run_details":details}
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(errors,indent=2),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe",type=Path)
    parser.add_argument("results",type=Path)
    parser.add_argument("source_figure",type=Path)
    args = parser.parse_args()
    compare(args.exe.resolve(),args.results,args.source_figure)
