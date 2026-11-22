"""Build distance bounds for shared STEP edges, in the model's millimetres."""
from collections import defaultdict

import numpy as np
from numba import njit


def bounded_chords(arc, tolerance):
    """Subdivide rational Bezier arcs until the control hull fits a chord capsule."""
    w = np.array([arc.Weight(i) for i in range(1,arc.NbPoles()+1)])
    xyz = np.array([arc.Pole(i).Coord() for i in range(1,arc.NbPoles()+1)])
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(w) & (w>0)):
        raise ValueError('Invalid rational control net')
    nets = np.column_stack((xyz*w[:,None],w))[None,:,:]
    chords = []
    for _ in range(32):
        pts = nets[:,:,:3]/nets[:,:,3:]
        a, b = pts[:,0], pts[:,-1]
        u = b-a
        sq = np.einsum('ij,ij->i',u,u)
        v = pts-a[:,None,:]
        param = np.einsum('ijk,ik->ij',v,u)/np.where(sq>0,sq,1.)[:,None]
        radius = np.linalg.norm(v-np.clip(param,0,1)[:,:,None]*u[:,None,:],axis=2).max(axis=1)
        accept = radius <= tolerance
        if accept.any():
            chords.append(np.stack((a[accept],b[accept]),axis=1))
        work = nets[~accept]
        if not len(work):
            return np.concatenate(chords)
        left, right = np.empty_like(work), np.empty_like(work)
        left[:,0],right[:,-1] = work[:,0],work[:,-1]
        for j in range(1,work.shape[1]):
            work = .5*(work[:,:-1]+work[:,1:])
            left[:,j],right[:,-1-j] = work[:,0],work[:,-1]
        nets = np.concatenate((left,right))
    raise ValueError('Unresolved STEP edge subdivision')


def build_index(step, field, tolerance_mm=1e-6):
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.Geom import Geom_TrimmedCurve
    from OCP.GeomConvert import GeomConvert, GeomConvert_BSplineCurveToBezierCurve
    from OCP.GeomAbs import (GeomAbs_Line,GeomAbs_Circle,GeomAbs_Ellipse,GeomAbs_Parabola,
                             GeomAbs_Hyperbola,GeomAbs_BezierCurve,GeomAbs_BSplineCurve)
    from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape
    from nhrep_gear import read_occt_solid
    from fitted_locator import pack_tree

    if not np.isfinite(tolerance_mm) or tolerance_mm<=0:
        raise ValueError('Expected positive chord tolerance')
    solid = read_occt_solid(step)
    faces, incidence = TopTools_IndexedMapOfShape(), TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapes_s(solid,TopAbs_FACE,faces)
    TopExp.MapShapesAndAncestors_s(solid,TopAbs_EDGE,TopAbs_FACE,incidence)
    pair_ids = np.sort(field.owner[field.edge_pairs],axis=1)
    needed = set(map(tuple,pair_ids))
    groups = defaultdict(list)
    unsupported = set()
    supported = (GeomAbs_Line,GeomAbs_Circle,GeomAbs_Ellipse,GeomAbs_Parabola,GeomAbs_Hyperbola,
                 GeomAbs_BezierCurve,GeomAbs_BSplineCurve)
    for i in range(1,incidence.Extent()+1):
        pair = tuple(sorted({faces.FindIndex(f) for f in incidence.FindFromIndex(i)}))
        if pair not in needed:
            continue
        edge = TopoDS.Edge_s(incidence.FindKey(i))
        if BRep_Tool.Degenerated_s(edge):
            unsupported.add(pair)
            continue
        adaptor = BRepAdaptor_Curve(edge)
        if adaptor.GetType() not in supported:
            unsupported.add(pair)
            continue
        curve = BRep_Tool.Curve_s(edge,0.,0.)
        trimmed = Geom_TrimmedCurve(curve,adaptor.FirstParameter(),adaptor.LastParameter())
        spans = GeomConvert_BSplineCurveToBezierCurve(GeomConvert.CurveToBSplineCurve_s(trimmed))
        for j in range(1,spans.NbArcs()+1):
            groups[pair].append(bounded_chords(spans.Arc(j),tolerance_mm))
    roots = {}
    boxes, children, segments = [], [], []
    nnode = nseg = depth = 0
    for pair in sorted(groups.keys()-unsupported):
        lines = np.concatenate(groups[pair])
        xyz = lines.reshape(-1,3)
        edges = np.arange(len(xyz)).reshape(-1,2)
        bounds,links,order,height = pack_tree(xyz,edges)
        # Weighted Bezier control hulls are inside the leaf chord capsules.
        # Outward padding includes the flattening tolerance and fp roundoff.
        pad = tolerance_mm + 256*np.finfo(float).eps*max(1.,np.max(abs(xyz)))
        bounds[:,0] -= pad
        bounds[:,1] += pad
        leaf = links[:,0]<0
        links[leaf,0] -= nseg
        links[leaf,1] += nseg
        links[~leaf] += nnode
        roots[pair] = nnode
        nnode += len(bounds)
        nseg += len(lines)
        depth = max(depth,height)
        boxes.append(bounds)
        children.append(links)
        segments.append(lines[order])
    root = np.array([roots.get(tuple(pair),-1) for pair in pair_ids],dtype=np.int32)
    return (root,np.concatenate(boxes) if boxes else np.empty((0,2,3)),
            np.concatenate(children) if boxes else np.empty((0,2),np.int32),
            np.concatenate(segments) if boxes else np.empty((0,2,3)),max(1,depth))


@njit(cache=True, inline='always')
def box_distance(point, box):
    sq = 0.
    for k in range(3):
        delta = max(box[0,k]-point[k], point[k]-box[1,k], 0.)
        sq += delta*delta
    return sq


@njit(cache=True, inline='always')
def segment_foot(point, a, b):
    dx, dy, dz = b[0]-a[0], b[1]-a[1], b[2]-a[2]
    length_sq = dx*dx + dy*dy + dz*dz
    u = 0.
    if length_sq:
        u = min(1., max(0., ((point[0]-a[0])*dx + (point[1]-a[1])*dy
                              + (point[2]-a[2])*dz) / length_sq))
    x, y, z = a[0]+u*dx, a[1]+u*dy, a[2]+u*dz
    sq = (point[0]-x)**2 + (point[1]-y)**2 + (point[2]-z)**2
    return sq, x, y, z


@njit(cache=True)
def edge_minimum(point, root, bounds, links, chords, stack, upper):
    tested_chords = 0  # Number of chords actually tested.
    best = upper*upper
    stack[0] = root
    pending = 1
    while pending:
        pending -= 1
        node = stack[pending]
        if box_distance(point,bounds[node]) > best:
            continue
        left,right = links[node]
        if left < 0:
            for j in range(~left,right):
                sq,_,_,_ = segment_foot(point,chords[j,0],chords[j,1])
                best = min(best,sq)
                tested_chords += 1
        else:
            if box_distance(point,bounds[left])<box_distance(point,bounds[right]):
                stack[pending],stack[pending+1] = right,left
            else:
                stack[pending],stack[pending+1] = left,right
            pending += 2
    return np.sqrt(best),tested_chords
