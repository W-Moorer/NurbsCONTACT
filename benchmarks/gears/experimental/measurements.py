"""Extract the NASA Fig. 7 traces and mesh-synchronous UNSW encoder TE."""

import argparse
import json
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat

from fetch import FILES


# Source coordinates are in the three-pixels-per-point render of PDF page 6.
# Hand-selected clicks follow the solid experimental trace through curve crossings.
# The last point in (d) is where the solid trace is still distinguishable.
NASA_PANELS = {
    "a": ((577, 861, 591, 248), [
        (609,591),(618,590),(624,582),(630,569),(640,550),(650,535),(660,512),
        (670,480),(680,450),(686,441),(692,439),(700,449),(708,464),(715,471),
        (723,468),(730,457),(737,441),(744,435),(751,436),(757,447),(764,455),
        (770,462),(778,460),(785,456),(791,458),(798,462),(806,466),(813,478),
        (820,495),(828,518),(836,542),(844,564),(850,579),(856,591),(860,591)
    ]),
    "b": ((916, 1202, 591, 246), [
        (963,591),(974,587),(982,579),(990,564),(1000,538),(1009,516),
        (1017,511),(1024,513),(1031,520),(1037,529),(1044,547),(1050,570),
        (1055,585),(1059,591),(1070,591),(1080,591),(1090,591),(1100,591),
        (1104,585),(1110,571),(1118,556),(1125,552),(1132,554),(1138,563),
        (1145,572),(1154,575),(1163,578),(1170,584),(1181,591),(1190,591)
    ]),
    "c": ((580, 864, 1053, 708), [
        (628,1053),(634,1050),(640,1038),(646,1018),(652,995),(658,973),
        (665,948),(673,925),(680,907),(688,892),(695,874),(702,867),(709,867),
        (716,873),(724,883),(731,894),(740,901),(752,901),(760,901),
        (768,906),(777,918),(787,939),(796,955),(804,971),(813,982),(821,986),
        (829,983),(836,990),(844,1015),(850,1034),(856,1052),(863,1053)
    ]),
    "d": ((918, 1203, 1053, 707), [
        (938,1053),(945,1050),(950,1041),(957,1023),(965,999),(973,974),
        (981,947),(991,920),(1000,889),(1010,859),(1019,830),(1028,805),
        (1038,782),(1048,761),(1055,746),(1061,744),(1067,749),(1075,758),
        (1083,765),(1090,768),(1097,766),(1103,765),(1108,769),(1113,779),
        (1118,790),(1124,801),(1130,815),(1137,837),(1144,857),(1152,873),
        (1160,896),(1168,919),(1176,942),(1182,952),(1188,968),(1195,982)
    ]),
}


def nasa(source, output, qa):
    with fitz.open(source / "nasa-tm-103232.pdf") as paper:
        pixmap = paper[5].get_pixmap(matrix=fitz.Matrix(3,3), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    proof = image.copy()
    mark = ImageDraw.Draw(proof)
    for key, (axes, points) in NASA_PANELS.items():
        x32, x12, y0, y2000 = axes
        pixels = np.array(points)
        roll = 32 - 20*(pixels[:,0]-x32)/(x12-x32)
        force = np.maximum(0,2000*(y0-pixels[:,1])/(y0-y2000))
        values = np.column_stack((roll, force, pixels))[::-1]
        np.savetxt(output/f"nasa-7{key}.csv", values, delimiter=",", comments="",
                   header="roll_deg,force_N,source_x_px,source_y_px", fmt="%.9g")
        for x, y in points:
            mark.ellipse((x-2,y-2,x+2,y+2),fill="#d82b55")
    proof.crop((480,230,1250,1150)).save(qa/"nasa-digitization.png")


def edge_times(signal, fs, threshold, rising=True):
    signal = signal if rising else -signal
    threshold = threshold if rising else -threshold
    crossing = np.flatnonzero((signal[:-1] < threshold) & (signal[1:] >= threshold))
    fraction = (threshold-signal[crossing])/(signal[crossing+1]-signal[crossing])
    return (crossing+fraction)/fs


def encoder_data(file):
    data = loadmat(file)
    fs = float(data["Fs"].item())
    signals = data["signals"]
    tor = float(data["torqueAvg"].item())
    tach = signals[:,0]
    scale = np.max(np.abs(tach))
    tach = tach if tach[np.argmax(np.abs(tach))]>0 else -tach
    rev = edge_times(tach,fs,.50*scale)
    rev = rev[np.r_[True,np.diff(rev)>.005]]
    if len(rev)<46:
        raise ValueError(f"Insufficient tachometer periods in {file}")
    speed = 1/np.median(np.diff(rev))
    tacho_blocks = np.arange(1,len(rev)-44,44)
    if len(tacho_blocks)==0:
        raise ValueError(f"No complete hunting-tooth periods in {file}")
    ratio = 27/44
    r1, r2 = np.cos(np.deg2rad(20))*np.array([27.,44.])
    cycles = 44*27
    sub = 128
    starts = 2*np.pi*np.arange(cycles*sub)/(27*sub)
    summaries = []
    edge_diagnostics = []
    counts = []
    for rising in (True,False):
        ei = edge_times(signals[:,1],fs,0.,rising)
        eo = edge_times(signals[:,3],fs,0.,rising)
        ai = 2*np.pi*np.arange(len(ei))/3600
        ao = 2*np.pi*np.arange(len(eo))/3600
        count = np.diff(np.searchsorted(ei,rev))
        counts.append(int(np.max(np.abs(count-3600))))
        if counts[-1]>1 or abs(len(eo)/len(ei)-ratio)>.0003:
            raise ValueError(f"Encoder count or gear ratio mismatch in {file}: {counts[-1]}")
        blocks = []
        for number in tacho_blocks:
            start = np.interp(rev[number],ei,ai)
            input_angle = start+starts
            time = np.interp(input_angle,ai,ei)
            output_angle = np.interp(time,eo,ao)
            if time[-1] >= eo[-1] or np.any(np.diff(time)<=0):
                raise ValueError(f"Encoder duration mismatch {file}")
            te = 1000*(output_angle*r2-input_angle*r1)
            average = te.reshape(cycles,sub).mean(axis=0)
            blocks.append(average-average.mean())
        blocks = np.array(blocks)
        summaries.append(blocks)
        edge_diagnostics.append([len(ei),len(eo)])
    wave = summaries[0].mean(axis=0)
    harmonics = (2*np.abs(np.fft.rfft(wave))/len(wave))[1:7]
    properties = {"source_file": file.name, "input_rpm": speed*60, "torque_Nm": tor, "sample_Hz": fs,
                  "blocks": len(tacho_blocks), "input_turns_per_block": 44, "mesh_cycles_per_block": cycles,
                  "input_edges_max_delta_per_revolution": counts, "edge_counts": edge_diagnostics,
                  "rising_falling_rms_um": float(np.sqrt(np.mean((wave-summaries[1].mean(axis=0))**2))),
                  "wave_rms_um": float(np.sqrt(np.mean(wave**2))), "peak_to_peak_um": float(np.ptp(wave)),
                  "amplitudes_um": harmonics.tolist()}
    phase = np.arange(sub)/sub
    return np.column_stack((phase,wave,summaries[0].min(axis=0),summaries[0].max(axis=0),
                            summaries[1].mean(axis=0))), properties


def process(source, output, qa):
    output.mkdir(parents=True,exist_ok=True)
    qa.mkdir(parents=True,exist_ok=True)
    nasa(source,output,qa)
    summaries = {}
    for file in FILES:
        values, stats = encoder_data(source/file)
        name=Path(file).stem
        np.savetxt(output/f"{name}.csv",values,delimiter=",",comments="",
                   header="mesh_cycle,te_um,block_min_um,block_max_um,falling_edge_um",fmt="%.10g")
        summaries[name]=stats
        print(name,stats,flush=True)
    (output/"unsw.json").write_text(json.dumps(summaries,indent=2)+"\n",encoding="utf-8")
    (output/"sources.json").write_text((source/"sources.json").read_text(encoding="utf-8"),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("previews",type=Path)
    args=parser.parse_args()
    process(args.source,args.output,args.previews)


if __name__ == "__main__":
    main()
