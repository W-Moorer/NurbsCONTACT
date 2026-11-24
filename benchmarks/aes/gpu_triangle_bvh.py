"""Static GPU triangle intersection query. The caller supplies device-resident relative poses."""
from pathlib import Path

import cupy as cp
import numpy as np


class TriangleMesh:
    """Pack a local-space AABB hierarchy in depth-first order with escape indices."""
    def __init__(self, vertices, faces, leaf_size=8):
        points=np.asarray(vertices,dtype=np.float64)
        faces=np.asarray(faces,dtype=np.int64)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
            raise ValueError("Expected finite 3D coordinates")
        if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or faces.min()<0 or faces.max()>=len(points):
            raise ValueError("Invalid triangles")
        if not 1<=leaf_size<=64: raise ValueError("Leaf size must be between 1 and 64")
        tri=points[faces]
        if np.any(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)==0):
            raise ValueError("Degenerate triangle")
        centers=tri.mean(axis=1)
        lo,hi=tri.min(axis=1),tri.max(axis=1)
        bounds,links,order=[],[],[]
        def area(extent):
            x,y,z=extent.T
            return x*y+y*z+z*x
        max_unbalanced_depth=2*int(np.ceil(np.log2(len(tri)+1)))
        def node(ids,depth):
            index=len(bounds)
            a,b=lo[ids].min(0),hi[ids].max(0)
            bounds.append(np.r_[a,b])
            links.append(None)
            first=len(order)
            if len(ids)<=leaf_size:
                order.extend(ids)
                count=len(ids)
            else:
                # Long cap facets and small curved facets can have nearly
                # coincident centroids. Minimize child box area times face count.
                best=np.inf
                for axis in range(3):
                    ranked_axis=ids[np.argsort(centers[ids,axis],kind="stable")]
                    lower,upper=lo[ranked_axis],hi[ranked_axis]
                    left=np.maximum.accumulate(upper,axis=0)-np.minimum.accumulate(lower,axis=0)
                    right=(np.maximum.accumulate(upper[::-1],axis=0)-np.minimum.accumulate(lower[::-1],axis=0))[::-1]
                    sizes=np.arange(1,len(ids))
                    costs=area(left[:-1])*sizes+area(right[1:])*(len(ids)-sizes)
                    split=int(np.argmin(costs))
                    if costs[split]<best:
                        best=costs[split]
                        ranked,middle=ranked_axis,split+1
                if depth>=max_unbalanced_depth: middle=len(ids)//2
                node(ranked[:middle],depth+1)
                node(ranked[middle:],depth+1)
                count=0
            links[index]=(len(bounds),first,count)
        node(np.arange(len(tri)),0)
        self.triangles=cp.asarray(np.ascontiguousarray(tri[order]))
        self.ids=cp.asarray(order,dtype=cp.int32)
        self.packed_indices=cp.asarray(np.argsort(order),dtype=cp.int32)
        self.bounds=cp.asarray(bounds,dtype=cp.float64)
        self.links=cp.asarray(links,dtype=cp.int32)
        self.leaves=cp.asarray([i for i,v in enumerate(links) if v[2]],dtype=cp.int32)
        self.count=len(tri)


class IntersectionQuery:
    """Return all intersecting original triangle IDs. Work buffers may be reused at the next pose."""
    def __init__(self, first, second, capacity=8192):
        if capacity<=0: raise ValueError("Expected positive hit capacity")
        self.first,self.second=first,second
        self.pairs=cp.empty((capacity,2),dtype=cp.int32)
        self.count=cp.zeros(2,dtype=cp.int32)
        self.candidates=cp.empty((max(8192,capacity*4),2),dtype=cp.int32)
        src=Path(__file__).resolve().parents[2]/"src/gpu_triangle_intersections.cu"
        module=cp.RawModule(code=src.read_text(),options=("--std=c++14","--fmad=false"))
        self.broad=module.get_function("leaf_candidates")
        self.narrow=module.get_function("intersect_leaves")
        self.geometry=module.get_function("intersection_geometry")
        self.geometry_buffers=None

    def enqueue(self, relative, *, geometry=False):
        """Query a device 3x4 relative pose. Optionally return sizes, vertices and face normals."""
        if relative.dtype!=cp.float64 or relative.shape!=(3,4) or not relative.flags.c_contiguous:
            raise ValueError("Expected a contiguous FP64 3 by 4 device transform")
        a,b=self.first,self.second
        self.count.fill(0)
        self.broad(((len(a.leaves)+127)//128,),(128,),
                   (a.leaves,np.int32(len(a.leaves)),a.bounds,b.bounds,b.links,np.int32(len(b.links)),
                    relative,self.candidates,np.int32(len(self.candidates)),self.count))
        self.narrow((128,),(128,),
                    (a.triangles,a.ids,a.links,b.triangles,b.ids,b.links,relative,
                     self.candidates,np.int32(len(self.candidates)),self.pairs,np.int32(len(self.pairs)),self.count))
        if geometry: return self._construct(relative)

    def read(self):
        n,leaves=self.count.get()
        if leaves>len(self.candidates): raise OverflowError(f"Query needs {leaves} leaf slots")
        if n>len(self.pairs): raise OverflowError(f"Query needs {n} hit slots")
        return self.pairs[:n].get()

    def _construct(self, relative):
        if self.geometry_buffers is None:
            n=len(self.pairs)
            self.geometry_buffers=(cp.empty(n,dtype=cp.int32),cp.empty((n,6,3)),cp.empty((n,2,3)))
        a,b=self.first,self.second
        self.geometry(((len(self.pairs)+127)//128,),(128,),
                      (a.triangles,a.packed_indices,b.triangles,b.packed_indices,relative,self.pairs,self.count,
                       np.int32(len(self.pairs)),*self.geometry_buffers))
        return self.geometry_buffers
