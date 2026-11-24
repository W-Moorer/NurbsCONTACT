"""Frozen-contact normal-mode frequencies from native contact points and the modeled shaft inertias."""
import argparse
import json
from pathlib import Path
import numpy as np


def frequencies(deck, points, out):
    with np.load(deck/"assembly.npz") as model:
        inertia,spring=model["coord"][:,0:2].T
        bodies,pairs,mapping=model["bodies"],model["pairs"],model["mapping"]
        tangent=np.asarray(model["stiffness"])
    records={}
    for j,(a,b) in enumerate(pairs):
        with next(points.glob(f"*_Mesh{j}@*.con")).open() as stream:
            next(stream)
            for line in stream:
                v=np.fromstring(line,sep=" ")
                if len(v) in (2,24): t=float(v[0]);v=v[2:]
                if not .102-1e-10<=t<=.114+1e-10: continue
                patch=records.setdefault(t,{}).setdefault(j,[])
                if len(v) and v[20]>0:
                    q=np.zeros(len(inertia))
                    for sign,i in [(-1,a),(1,b)]:
                        lever=(v[7:10]-bodies[i,4:7])*1e-3
                        q[mapping[i]]+=sign*np.dot(v[10:13],np.cross(bodies[i,1:4],lever))
                    patch.append(q)
    f=[]
    for t,patches in sorted(records.items()):
        if len(patches)!=len(pairs): raise ValueError("Missing stage")
        k=np.diag(spring.copy())
        for j,patch in patches.items():
            if patch:
                J=np.array(patch)
                k+=tangent[j]/len(J)*(J.T@J)
        eig=np.linalg.eigvalsh(k/np.sqrt(inertia[:,None]*inertia[None,:]))
        f.append(np.sqrt(max(eig[-1],0))/(2*np.pi))
    quantiles=np.percentile(f,[0,50,100])
    result=dict(frames=len(f),highest_mode_Hz_min_median_max=quantiles.tolist(),
                samples_per_median_period={str(dt):float(1/(dt*1e-6*quantiles[1])) for dt in (100,50,25)})
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("deck","points","out"):p.add_argument(key,type=Path)
    frequencies(**vars(p.parse_args()))
