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
        self.proxy=self.triangles.astype(cp.float32)
        self.extent=float(np.max(np.abs(points)))
        self.ids=cp.asarray(order,dtype=cp.int32)
        self.packed_indices=cp.asarray(np.argsort(order),dtype=cp.int32)
        self.bounds=cp.asarray(bounds,dtype=cp.float64)
        self.links=cp.asarray(links,dtype=cp.int32)
        # Round bounds outwards. Two float4 loads carry both bounds and traversal links.
        boxes=np.asarray(bounds)
        packed=np.empty((len(boxes),2,4),dtype=np.float32)
        packed[:,0,:3]=np.nextafter(boxes[:,:3].astype(np.float32),-np.float32(np.inf))
        packed[:,1,:3]=np.nextafter(boxes[:,3:].astype(np.float32),np.float32(np.inf))
        integers=packed.view(np.int32)
        integers[:,0,3]=np.asarray(links)[:,0]
        integers[:,1,3]=np.asarray(links)[:,2]
        self.traversal=cp.asarray(packed)
        leaf_ids=np.flatnonzero(np.asarray(links)[:,2]).astype(np.int32)
        self.leaves=cp.asarray(leaf_ids)
        self.count=len(tri)
        self.leaf_size=leaf_size


class IntersectionQuery:
    """Return all intersecting original triangle IDs. Work buffers may be reused at the next pose."""
    def __init__(self, first, second, capacity=8192):
        if capacity<=0: raise ValueError("Expected positive hit capacity")
        self.first,self.second=first,second
        self.pairs=cp.empty((capacity,2),dtype=cp.int32)
        self.count=cp.zeros(2,dtype=cp.int32)
        self.candidates=cp.empty((max(8192,capacity*4),2),dtype=cp.int32)
        src=Path(__file__).with_name("kernels")/"gpu_triangle_intersections.cu"
        module=cp.RawModule(code=src.read_text(),options=("--std=c++14","--fmad=false"))
        self.module=module
        self.lanes_per_leaf_pair=min(32,1<<(int(first.leaf_size*second.leaf_size)-1).bit_length())
        self.broad=module.get_function("leaf_candidates")
        self.narrow=module.get_function("intersect_leaves")
        self.geometry=module.get_function("intersection_geometry")
        self.geometry_buffers=None

    def enqueue(self, relative, *, geometry=False):
        """Query a device 3x4 relative pose. Optionally return sizes, vertices and face normals."""
        self.enqueue_candidates(relative)
        self.enqueue_intersections(relative)
        if geometry: return self._construct(relative)

    def enqueue_candidates(self, relative):
        """Update the candidate leaf pairs from the current source-to-target transform."""
        if relative.dtype!=cp.float64 or relative.shape!=(3,4) or not relative.flags.c_contiguous:
            raise ValueError("Expected a contiguous FP64 3 by 4 device transform")
        a,b=self.first,self.second
        self.count.fill(0)
        self.broad(((len(a.leaves)+127)//128,),(128,),
                   (a.leaves,np.int32(len(a.leaves)),a.bounds,b.traversal,np.int32(len(b.links)),
                    relative,self.candidates,np.int32(len(self.candidates)),self.count))
    def enqueue_intersections(self, relative):
        """Test the current candidates at the same transform, retaining every hit pair."""
        a,b=self.first,self.second
        self.narrow((128,),(128,),
                    (a.triangles,a.ids,a.links,b.triangles,b.ids,b.links,relative,
                     a.proxy,b.proxy,np.float64(a.extent+b.extent),np.int32(self.lanes_per_leaf_pair),
                     self.candidates,np.int32(len(self.candidates)),self.pairs,np.int32(len(self.pairs)),self.count))

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


class IntersectionBatch:
    """Enumerate all triangle intersections for independent body pairs at one instant.

    Supply one source-to-target rigid transform per pair. Meshes can be shared
    across pairs. Workspaces and counts stay on the device.
    """

    def __init__(self, mesh_pairs, capacity=8192, *, reuse_distance=0.0, fine_lanes=8):
        if not np.isfinite(reuse_distance) or reuse_distance < 0:
            raise ValueError("reuse distance must be finite and nonnegative")
        self.skin = np.float64(reuse_distance)
        if fine_lanes not in (1, 4, 8, 16, 32):
            raise ValueError("fine_lanes must be 1, 4, 8, 16 or 32")
        self.fine_lanes = fine_lanes
        self.queries = [IntersectionQuery(a, b, capacity) for a, b in mesh_pairs]
        if not self.queries:
            raise ValueError("at least one mesh pair is required")
        pointers = (
            "leaves",
            "bounds",
            "traversal",
            "a",
            "a_ids",
            "a_links",
            "b",
            "b_ids",
            "b_links",
            "a_proxy",
            "b_proxy",
            "candidates",
            "hits",
            "filtered",
        )
        sizes = ("leaves_n", "nodes", "candidate_n", "hit_n", "width", "filtered_n")
        layout = np.dtype(
            [(key, np.uintp) for key in pointers]
            + [("extent", np.float64)]
            + [(key, np.int32) for key in sizes],
            align=True,
        )
        views = np.empty(len(self.queries), dtype=layout)
        self._counts = cp.zeros(3 * len(views), dtype=cp.int32)
        self.counts = self._counts[: 2 * len(views)].reshape(len(views), 2)
        self.filtered_counts = self._counts[2 * len(views) :]
        self.anchors = cp.full((len(views), 3, 4), cp.nan)
        self.refresh = cp.empty(len(views), dtype=cp.int32)
        self._filtered = []
        for i, q in enumerate(self.queries):
            q.count = self.counts[i]
            a, b = q.first, q.second
            filtered = cp.empty_like(q.candidates if self.skin else q.pairs)
            self._filtered.append(filtered)
            fields = (
                a.leaves,
                a.bounds,
                b.traversal,
                a.triangles,
                a.ids,
                a.links,
                b.triangles,
                b.ids,
                b.links,
                a.proxy,
                b.proxy,
                q.candidates,
                q.pairs,
                filtered,
            )
            views[i] = (
                *(v.data.ptr for v in fields),
                a.extent + b.extent,
                len(a.leaves),
                len(b.links),
                len(q.candidates),
                len(q.pairs),
                q.lanes_per_leaf_pair,
                len(filtered),
            )
        self.views = cp.asarray(
            views.view(np.uint8).reshape(len(views), layout.itemsize)
        )
        self.blocks = (max(len(q.first.leaves) for q in self.queries) + 127) // 128
        module = self.queries[0].module
        self.check_cache = module.get_function("check_candidate_cache")
        self.broad = module.get_function("batch_leaf_candidates")
        self.filter = module.get_function("batch_filter")
        self.narrow = module.get_function("batch_intersections")
        self.group_narrow = module.get_function("batch_group_intersections")

    def _launch(self, kernel, blocks, transforms, *arrays, skin=False):
        for first in range(0, len(self.queries), 65535):
            count = min(65535, len(self.queries) - first)
            args = (self.views[first:], transforms[first:], self.counts[first:])
            args += tuple(array[first:] for array in arrays)
            if skin:
                args += (self.skin,)
            kernel((blocks, count), (128,), args)

    def enqueue_candidates(self, transforms):
        if (
            transforms.shape != (len(self.queries), 3, 4)
            or transforms.dtype != cp.float64
            or not transforms.flags.c_contiguous
        ):
            raise ValueError(
                "expected one contiguous device FP64 3 by 4 transform per pair"
            )
        if self.skin:
            self._launch(
                self.check_cache,
                1,
                transforms,
                self.filtered_counts,
                self.refresh,
                self.anchors,
                skin=True,
            )
        else:
            self._counts.fill(0)
        self._launch(self.broad, self.blocks, transforms, self.refresh, skin=True)

    def enqueue_intersections(self, transforms):
        self._launch(
            self.filter, 128, transforms, self.filtered_counts, self.refresh, skin=True
        )
        if self.fine_lanes == 1:
            self._launch(self.narrow, 64, transforms, self.filtered_counts, skin=True)
        else:
            for first in range(0, len(self.queries), 65535):
                count = min(65535, len(self.queries) - first)
                self.group_narrow((64, count), (128,),
                    (self.views[first:], transforms[first:], self.counts[first:], self.filtered_counts[first:],
                     self.skin, np.int32(self.fine_lanes)))

    def enqueue(self, transforms, *, geometry=False):
        self.enqueue_candidates(transforms)
        self.enqueue_intersections(transforms)
        if geometry:
            return [q._construct(t) for q, t in zip(self.queries, transforms)]
