"""Decode the saved videos and render the supplement pages for visual inspection."""

import argparse
import json
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"hht"))
from verify_artifacts import sheet, verify_video


def paper(root,target):
    pages=[]
    with fitz.open(root/"paper"/"industrial.pdf") as doc:
        text="".join(page.get_text() for page in doc)
        for phrase in ("Cylindrical Roller Bearing","NASA","150,000"):
            if phrase not in text or "??" in text:
                raise ValueError(f"Missing resolved supplement content {phrase}")
        for i,page in enumerate(doc,start=1):
            pixels=page.get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False)
            img=Image.frombytes("RGB",(pixels.width,pixels.height),pixels.samples)
            img.save(target/f"paper-{i}.png")
            pages.append((f"Page {i}",img))
    sheet(pages,(298,421),3,target/"paper.png")
    return len(pages)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    parser.add_argument("previews",type=Path)
    args=parser.parse_args()
    args.previews.mkdir(exist_ok=True,parents=True)
    videos=[]
    panels=[]
    views={}
    for root,expected in ((args.results,8),(args.results/"bearing",2)):
        entries=json.loads((root/"animations"/"index.json").read_text(encoding="utf-8"))
        assert len(entries)==expected
        for entry in entries:
            frames,stats=verify_video(root,entry)
            videos.append(stats)
            views.setdefault(entry["case"],[]).append(np.asarray(frames[0],dtype=float)[83:-9])
            for name,frame in zip(("start","one third","two thirds","end"),frames,strict=True):
                panels.append((f"{entry['case']} / {entry['view']} / {name}",frame))
    for scene,frames in views.items():
        assert len(frames)==2 and np.mean(np.abs(frames[0]-frames[1]))>.25,scene
        assert (args.results/"assemblies"/f"{scene}.step").is_file(),scene
    sheet(panels,(290,163),4,args.previews/"animations.png")
    pictures=[]
    gear_folder=args.results/"figures"
    bearing_folder=args.results/"bearing"/"figures"
    for folder,names in ((gear_folder,("gear-models","gear-dynamics","gear-cost-accuracy",
                                         "gear-axial-refinement","reducer-duty")),
                         (bearing_folder,("bearing-loads",))):
        for name in names:
            with fitz.open(folder/f"{name}.pdf") as figure:
                assert len(figure)==1,name
            with Image.open(folder/f"{name}.png") as image:
                pictures.append((name,image.copy()))
    sheet(pictures,(505,390),2,args.previews/"figures.png")
    results={"paper_pages":paper(args.results,args.previews),"videos":videos}
    (args.previews/"verification.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print(f"Validated {len(videos)} moving videos, {len(pictures)} figures and {results['paper_pages']} pages")


if __name__=="__main__":
    main()
