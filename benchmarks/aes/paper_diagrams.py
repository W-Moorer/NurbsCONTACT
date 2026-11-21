"""Draw the three method diagrams and the workflow used in the AES paper."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle
import numpy as np


INK = "#1b2c3e"
BLUE = "#2d6595"
GREEN = "#187c72"
AMBER = "#aa5d1b"
PALE = "#edf3f8"
GRID = "#dce7ee"

plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "mathtext.fontset":"stix",
                     "pdf.fonttype":42, "text.color":INK, "axes.labelcolor":INK,
                     "xtick.color":INK, "ytick.color":INK, "axes.edgecolor":INK})


def save(fig, output, name):
    fig.savefig(output/f"{name}.pdf", bbox_inches="tight", facecolor="white", pad_inches=.06)
    plt.close(fig)


def interval(ax, lower, upper, y, color):
    ax.plot([lower,upper],[y,y],color=color,lw=2.6,solid_capstyle="round")
    ax.plot([lower,upper],[y,y],"|",color=color,ms=12,mew=2)


def residual(output):
    fig,ax = plt.subplots(figsize=(7.25,2.66),layout="constrained")
    ax.set_xbound(0,3.18)
    ax.set_ybound(-.45,2.56)
    ax.axvspan(0,1.34,color=PALE,zorder=-3)
    ax.axvline(1.34,color=GREEN,lw=1.2,ls="--")
    ax.text(1.32,2.42,r"convergence threshold $\tau$",color=GREEN,ha="right")
    ax.text(.61,2.1,"Acceptable residual",ha="center",color=GREEN,weight="bold")
    for x0,x1,y,color in ((.93,1.83,1.52,AMBER),(1.08,1.23,.76,GREEN),(1.91,2.84,0,BLUE)):
        interval(ax,x0,x1,y,color)
    ax.text(1.91,1.52,"Refine contact force",va="center",color=AMBER)
    ax.text(.99,.76,"Accept",va="center",ha="right",color=GREEN)
    ax.text(1.83,0,"Continue Newton",va="center",ha="right",color=BLUE)
    ax.annotate("",xy=(1.16,.95),xytext=(1.29,1.32),arrowprops={"arrowstyle":"->","color":INK,"lw":.85})
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(r"Admissible interval for $\|R\|_\infty$")
    for side in ("top","left","right"):
        ax.spines[side].set_visible(False)
    save(fig,output,"principle-residual")


def event(output):
    fig,(ax,timeline) = plt.subplots(2,1,figsize=(7.25,3.42),gridspec_kw={"height_ratios":[3,1]})
    fig.subplots_adjust(hspace=.03)
    ax.set(xlim=(0,8),ylim=(0,2.9),aspect="equal")
    ax.axis("off")
    ax.add_patch(Rectangle((.55,.6),2.24,.33,facecolor=GRID,edgecolor=BLUE,linewidth=1.2))
    ax.add_patch(Rectangle((5.3,.6),2.15,.33,facecolor=GRID,edgecolor=BLUE,linewidth=1.2))
    ax.text(6.52,.34,"Trimmed stationary face",ha="center")
    ax.plot([2.79,5.30],[.93,.93],color=GRID,ls=":",lw=1.)
    ax.text(3.7,.39,"Opening",ha="center",color=BLUE)
    for y,color in ((2.12,GRID),(1.58,AMBER),(1.23,GREEN)):
        ax.plot([3.24,4.88],[y,y],color=color,lw=3.1,solid_capstyle="round")
    ax.annotate("Relative motion",xy=(3.44,1.36),xytext=(1.51,2.08),
                arrowprops={"arrowstyle":"->","color":INK},ha="center")
    ax.annotate("Projected CAD witness",xy=(4.88,1.23),xytext=(5.21,2.18),
                arrowprops={"arrowstyle":"->","color":GREEN},color=GREEN)
    ax.plot([4.88,5.30],[1.23,.93],"o--",color=GREEN,ms=4,lw=1.2)
    ax.text(5.48,1.30,r"$d^+\leq\delta$",ha="left",color=GREEN)
    timeline.set(xlim=(0,8),ylim=(0,1),xticks=[],yticks=[])
    timeline.axis("off")
    timeline.annotate("",xy=(7.91,.49),xytext=(.20,.49),arrowprops={"arrowstyle":"->","color":INK})
    timeline.plot([.23,3.46],[.49,.49],color=BLUE,lw=4,solid_capstyle="butt")
    timeline.axvspan(3.46,5.37,ymin=.35,ymax=.65,color=PALE)
    interval(timeline,3.46,5.37,.49,GREEN)
    timeline.text(1.67,.75,"Proved clear prefix",color=BLUE,ha="center")
    timeline.text(4.42,.75,"First-entry bracket",color=GREEN,ha="center")
    timeline.text(3.46,.08,r"$t_-$",ha="center")
    timeline.text(5.37,.08,r"$t_+$",ha="center")
    timeline.text(7.73,.09,"Time",ha="right")
    save(fig,output,"principle-event")


def reuse(output):
    fig,ax = plt.subplots(figsize=(7.25,2.67),layout="constrained")
    ax.set(xlim=(0,4.25),ylim=(0,2.82),xticks=[],yticks=[])
    x = np.linspace(0,4.23,180)
    ax.fill_between(x,0,2.46-.5*x,where=x<=2.64,color=PALE,zorder=-3)
    ax.plot(x,2.46-.5*x,color=BLUE,lw=2.0,label=r"Transported lower bound $L_0-\Delta_{AB}$")
    ax.axhline(1.14,color=AMBER,ls="--",lw=1.2)
    ax.plot([2.64,2.64],[0,1.14],color=INK,ls=":",lw=.9)
    ax.text(4.17,1.25,r"Contact activation $s$",ha="right",color=AMBER)
    ax.text(.26,2.57,r"Stored lower bound $L_0$",color=BLUE)
    ax.plot([.82,3.31],[2.05,.805],"o",color=BLUE,ms=5)
    ax.text(.81,1.69,"Skip pair",ha="center",color=GREEN,weight="bold")
    ax.text(3.31,.50,"Query again",ha="center",color=AMBER,weight="bold")
    ax.set_xlabel(r"Upper bound on relative point travel $\Delta_{AB}$")
    ax.set_ylabel("Separation")
    for side in ("top","right"):
        ax.spines[side].set_visible(False)
    save(fig,output,"principle-reuse")


def workflow(output):
    fig,ax=plt.subplots(figsize=(7.25,3.45))
    ax.set(xlim=(0,12),ylim=(0,5.7))
    ax.axis("off")

    def node(x,y,w,h,label,color=BLUE):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.12,rounding_size=.12",
                                    facecolor=PALE,edgecolor=color,lw=1.15))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=8.9,color=color)

    def arrow(a,b,color=INK,style="arc3"):
        ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->","color":color,"lw":1.1,"connectionstyle":style})

    node(3.74,4.77,4.51,.65,"STEP faces, trim curves and point-travel bounds")
    node(.23,3.38,3.21,.73,"HHT trial state\nRequest force tolerance")
    node(4.21,3.38,3.58,.73,"Bound contact loads\nRevalidate cached exclusions")
    node(8.53,3.38,3.25,.73,"Prescribed rigid path\nSet a clearance threshold")
    node(.23,1.82,3.21,.73,"Accept step or subdivide",GREEN)
    node(4.21,1.82,3.58,.73,"Test residual interval\nRefine ambiguous decisions")
    node(8.53,1.82,3.25,.73,"Exclude swept cells\nConstruct CAD witness")
    node(8.53,.23,3.25,.73,"Clear path, entry bracket\nor safe prefix",GREEN)
    arrow((5.5,4.65),(5.5,4.25))
    arrow((7.61,4.67),(9.45,4.24))
    arrow((3.57,3.75),(4.09,3.75))
    arrow((6.0,3.26),(6.0,2.67))
    arrow((4.09,2.20),(3.56,2.20),GREEN)
    arrow((7.81,2.37),(7.81,3.55),AMBER,"arc3,rad=.49")
    ax.text(5.99,1.51,"Repeat on an ambiguous decision",color=AMBER,ha="center",fontsize=8.6)
    arrow((10.15,3.26),(10.15,2.67))
    arrow((10.15,1.70),(10.15,1.09),GREEN)
    save(fig,output,"workflow")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for draw in (residual,event,reuse,workflow):
        draw(args.output)


if __name__ == "__main__":
    main()
