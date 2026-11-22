"""Compare STEP crease bounds and triangle backends at a common point-distance tolerance."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np


MODEL = Path('results/aes/gear-field')
STEP = Path('results/gears/models/spur-1a.step')


def mesh_step(deflection_um, interior, parallel=False, step=STEP, face_ids=None, visit_face=None):
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IMeshTools import IMeshTools_Parameters
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from nhrep_gear import read_occt_solid

    solid = read_occt_solid(step)
    params = IMeshTools_Parameters()
    params.Relative = False
    params.Deflection = deflection_um/1000
    params.Angle = .1
    params.ControlSurfaceDeflection = interior
    params.InParallel = parallel
    mesher = BRepMesh_IncrementalMesh(solid,params)
    if not mesher.IsDone():
        raise RuntimeError('STEP tessellation failed')
    vertices, faces, offset = [], [], 0
    if face_ids is not None:
        indexed = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(solid,TopAbs_FACE,indexed)
    it = TopExp_Explorer(solid,TopAbs_FACE)
    while it.More():
        face, location = TopoDS.Face_s(it.Current()), TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face,location)
        tr = location.Transformation()
        v = np.array([mesh.Node(i).Transformed(tr).Coord() for i in range(1,mesh.NbNodes()+1)])
        f = np.array([t.Get() for t in mesh.Triangles()],np.int32)-1
        if face.Orientation()==TopAbs_REVERSED:
            f = f[:,[0,2,1]]
        vertices.append(v)
        faces.append(f+offset)
        if face_ids is not None:
            face_ids.append(np.full(len(f), indexed.FindIndex(face), dtype=np.int32))
        if visit_face is not None:
            visit_face(face, location, mesh, v, f)
        offset += len(v)
        it.Next()
    v, f = np.concatenate(vertices), np.concatenate(faces)
    _, ids, inv = np.unique(np.rint(v/1e-6).astype(np.int64),axis=0,return_index=True,return_inverse=True)
    return v[ids],inv[f].astype(np.uint32)


def shorten_edges(v, f, length_mm):
    """Conforming midpoint bisection; all child facets lie in their parent plane."""
    if not np.isfinite(length_mm) or length_mm<=0:
        raise ValueError('Expected a positive maximum edge length')
    while True:
        end = np.roll(f,-1,axis=1)
        split = np.linalg.norm(v[f]-v[end],axis=2)>length_mm
        count = split.sum(axis=1)
        if not np.any(count):
            return v,f
        a, b = f[split], end[split]
        keys = (np.minimum(a,b).astype(np.uint64)<<32) | np.maximum(a,b)
        unique, inv = np.unique(keys,return_inverse=True)
        if len(v)+len(unique) >= 2**32:
            raise ValueError('Mesh exceeds the triangle backend index range')
        mids = np.zeros_like(f)
        mids[split] = len(v)+inv
        v = np.concatenate((v,(v[unique>>32]+v[unique & 0xffffffff])*.5))
        pieces = [f[count==0]]
        for n in (1,2):
            mask = count==n
            # Put the lone split (or unsplit) edge first in cyclic order.
            j = np.argmax(split[mask] if n==1 else ~split[mask],axis=1)
            order = (j[:,None]+np.arange(3))%3
            a,b,c = np.take_along_axis(f[mask],order,axis=1).T
            ab,bc,ca = np.take_along_axis(mids[mask],order,axis=1).T
            if n==1:
                pieces.extend((np.c_[a,ab,c],np.c_[ab,b,c]))
            else:
                pieces.append(np.c_[ca,bc,c])
                shorter = np.linalg.norm(v[a]-v[bc],axis=1)<=np.linalg.norm(v[b]-v[ca],axis=1)
                for k, triangles in [(shorter,((a,b,bc),(a,bc,ca))),(~shorter,((a,b,ca),(b,bc,ca)))]:
                    pieces.extend(np.column_stack(t)[k] for t in triangles)
        mask = count==3
        a,b,c = f[mask].T
        ab,bc,ca = mids[mask].T
        pieces.extend((np.c_[a,ab,ca],np.c_[ab,b,bc],np.c_[ca,bc,c],np.c_[ab,bc,ca]))
        f = np.concatenate(pieces)


def samples():
    blocks, targets = {}, {}
    for seed, folder in [(1407,'build/aes-gear-field'),(5119,'build/aes-gear-field-fast')]:
        blocks[str(seed)] = np.loadtxt(Path(folder)/f'probe-{seed}.txt')
        targets[str(seed)] = np.loadtxt(Path(folder)/f'truth-{seed}.csv',delimiter=',',skiprows=1,usecols=0)
    with (MODEL/'transition-points.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    blocks['contact'] = np.array([[float(row[k+'_mm']) for k in 'xyz'] for row in rows])
    targets['contact'] = np.array([float(row['cad_distance_mm']) for row in rows])
    return blocks,targets,rows


def quality(predict, blocks, targets, rows):
    result = {}
    for key,x in blocks.items():
        d = predict(x)
        truth = targets[key]
        if not np.isfinite(d).all():
            raise ValueError('Nonfinite signed distance')
        error = 1000*abs(d-truth)
        result[key] = {'n':len(x),'within':int(sum(error<=.125)),'max_um':float(max(error)),
                       'wrong_sign':int(sum(np.sign(d)!=np.sign(truth)))}
        if key=='contact':
            by_pair = {}
            for row,delta in zip(rows,np.maximum(.055-d,0)-np.maximum(.055-truth,0),strict=True):
                pair = (row['time_s'],row['face_pair'])
                by_pair[pair] = by_pair.get(pair,0.)+1e6*float(row['weight_m'])*delta
            result[key]['max_line_force_error_N'] = max(map(abs,by_pair.values()))
    return result


def mesh_queries(v, f, double):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)),o3d.core.Tensor(f.astype(np.uint32)))
    def mesh(x):
        return scene.compute_signed_distance(o3d.core.Tensor(x.astype(np.float32)),nthreads=1,nsamples=3).numpy()
    if not double:
        return mesh
    import igl
    tri = f.astype(np.int64)
    tree = igl.AABB()
    tree.init(v,tri)
    def mixed(x):
        sq,_,_ = tree.squared_distance(v,tri,x)
        inside = scene.compute_occupancy(o3d.core.Tensor(x.astype(np.float32)),nthreads=1,nsamples=3).numpy()
        return np.where(inside>.5,-1.,1.)*np.sqrt(sq)
    return mixed


def check_closed(v, f):
    """Verify oppositely directed edge pairs and positive oriented volume."""
    keys = np.empty(3*len(f),np.uint64)
    for i in range(3):
        a,b = f[:,i],f[:,(i+1)%3]
        if np.any(a==b):
            raise ValueError('Collapsed triangle edge')
        edge = np.minimum(a,b).astype(np.uint64)*len(v)+np.maximum(a,b).astype(np.uint64)
        keys[i*len(f):(i+1)*len(f)] = edge*2+(a>b)
    keys.sort()
    if len(keys)%2 or np.any(keys[::2]%2) or np.any(keys[::2]+1!=keys[1::2]) or np.any(keys[2::2]==keys[:-2:2]):
        raise ValueError('Open, inconsistently oriented or non-manifold triangle mesh')
    volume = 0.
    for i in range(0,len(f),65536):
        a,b,c = v[f[i:i+65536]].transpose(1,0,2)
        volume += np.einsum('ij,ij->',a,np.cross(b,c))/6
    if volume<=0:
        raise ValueError('Expected an outward-oriented mesh')
    return float(volume)


def direct_edge_check(field,index,blocks):
    """Compare the chord index with independent OCCT projections onto trimmed curves."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
    from OCP.TopAbs import TopAbs_EDGE,TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape,TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Pnt
    from fitted_locator import locate
    from trim_bounds import edge_minimum
    from nhrep_gear import read_occt_solid
    solid = read_occt_solid(STEP)
    faces = TopTools_IndexedMapOfShape()
    incidence = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapes_s(solid,TopAbs_FACE,faces)
    TopExp.MapShapesAndAncestors_s(solid,TopAbs_EDGE,TopAbs_FACE,incidence)
    curves = {}
    for i in range(1,incidence.Extent()+1):
        pair = tuple(sorted({faces.FindIndex(f) for f in incidence.FindFromIndex(i)}))
        edge = TopoDS.Edge_s(incidence.FindKey(i))
        if len(pair)==2 and not BRep_Tool.Degenerated_s(edge):
            a = BRepAdaptor_Curve(edge)
            curves.setdefault(pair,[]).append((BRep_Tool.Curve_s(edge,0.,0.),a.FirstParameter(),a.LastParameter()))
    roots,boxes,links,chords,depth = index
    stack = np.empty(depth,np.int32)
    errors = []
    for x in blocks.values():
        _,_,_,edges,_ = locate(x,field.xyz,field.tri,field.edges,field.bary_grad,field.tri_tree,
                               field.crease_tree,field.tri_grid,field.edge_grid,field.crease_band_mm)
        for i in np.flatnonzero(edges>=0):
            edge = edges[i]
            if roots[edge]<0:
                continue
            key = tuple(sorted(field.owner[field.edge_pairs[edge]]))
            pt = gp_Pnt(*x[i])
            exact = np.inf
            for curve,lo,hi in curves[key]:
                projector = GeomAPI_ProjectPointOnCurve(pt,curve,lo,hi)
                exact = min(exact,pt.Distance(curve.Value(lo)),pt.Distance(curve.Value(hi)))
                if projector.NbPoints():
                    exact = min(exact,projector.LowerDistance())
            got,_ = edge_minimum(x[i],roots[edge],boxes,links,chords,stack,np.inf)
            errors.append(abs(got-exact)*1000)
    if not errors or max(errors)>.0010001:
        raise AssertionError('Shared-edge query exceeds its chord tolerance')
    return {'n':len(errors),'max_error_um':max(errors)}


def compare(folder, repeats):
    from copy import copy
    from functools import partial
    from fitted_locator import locate, signed_distances
    from fitted_projection import refine
    from gear_field import FittedFaces
    from gear_field_replay import timed
    from trim_bounds import build_index
    import open3d as o3d

    blocks, targets, rows = samples()
    field = FittedFaces(MODEL/'guide.npz',MODEL/'faces.npz')
    start = perf_counter()
    index = build_index(STEP,field)
    report = {'repeats':repeats,'threads':1,'open3d':o3d.__version__,'threshold_um':.125,
              'chord_tolerance_um':.001,'trim_build_s':perf_counter()-start,
              'chords':len(index[3]),'unsupported_guide_edges':int(sum(index[0]<0)),
              'trim_index_MiB':sum(a.nbytes for a in index[:4])/2**20,'methods':{},'pruning':{}}
    report['direct_edge_check'] = direct_edge_check(field,index,blocks)
    methods = [('fitted_joint_original',field.predict,None)]
    for name, prune in [('trim_full',False),('trim_bounded_ours',True)]:
        variant = copy(field)
        variant.refine = partial(signed_distances,trim_index=index,prune=prune)
        methods.append((name,variant.predict,None))
    # Keep search and classification in the timer for every mesh backend.
    meshes = [('original_open3d',None,'open3d'),('compact_open3d','mesh-0.1','open3d'),
              ('compact_double_bvh_parity','mesh-0.1','igl_parity'),
              ('compact_double_pseudonormal','mesh-0.1','pseudonormal'),
              ('fine_double_pseudonormal','mesh-0.04','pseudonormal'),
              ('conditioned_4_open3d','conditioned-4','open3d'),
              ('conditioned_2_open3d','conditioned-2','open3d'),
              ('conditioned_1_open3d','conditioned-1','open3d')]
    for name, filename, backend in meshes:
        if filename is None:
            v,f = field.xyz,field.tri
        else:
            with np.load(folder/(filename+'.npz')) as archive:
                v,f = archive['v'],archive['f']
        start = perf_counter()
        if backend=='pseudonormal':
            from triangle_sdf import mesh_query
            predict = mesh_query(v,f)
        else:
            predict = mesh_queries(v,f,backend=='igl_parity')
        predict(blocks['contact'][:4])  # Open3D builds the scene lazily.
        report['methods'][name] = {'triangles':len(f),'vertices':len(v),
                                    'index_and_first_query_s':perf_counter()-start}
        methods.append((name,predict,None))
    for name,predict,_ in methods:
        report['methods'].setdefault(name,{})['quality'] = quality(predict,blocks,targets,rows)
    profile_blocks = dict(blocks)
    # Independent previously published regression inputs, outside the timing workload.
    for label in ('crease_band','bounding_box'):
        profile_blocks[label] = np.loadtxt(folder.parent/f'{label}-probes.txt')
    args = (field.xyz,field.tri,field.owner,field.uv,field.incident,field.edge_pairs,field.convex_sign,
            field.bary_grad,*field.packed,field.crease_band_mm)
    for key,x in profile_blocks.items():
        loc = locate(x,field.xyz,field.tri,field.edges,field.bary_grad,field.tri_tree,field.crease_tree,
                     field.tri_grid,field.edge_grid,field.crease_band_mm)
        counters = [np.zeros(3,np.int64) for _ in range(2)]
        full,bounded = [refine(x,*loc,*args,index,prune,c) for prune,c in zip((False,True),counters,strict=True)]
        np.testing.assert_allclose(full,bounded,rtol=0,atol=1e-12)
        np.testing.assert_allclose(bounded,methods[2][1](x),rtol=0,atol=1e-12)
        truth = targets[key] if key in targets else np.loadtxt(folder.parent/f'{key}-cad.csv',skiprows=1,delimiter=',',usecols=0)
        if np.any(abs(bounded-truth)>.000125) or np.any(np.sign(bounded)!=np.sign(truth)):
            raise AssertionError(f'CAD distance regression on {key}')
        report['pruning'][key] = {'wedge_queries':int(counters[1][0]),'skipped_edge':int(counters[1][1]),
                                  'chord_tests_full':int(counters[0][2]),'chord_tests_pruned':int(counters[1][2]),
                                  'full_vs_pruned_max_um':float(max(abs(full-bounded))*1000),
                                  'max_cad_error_um':float(max(abs(bounded-truth))*1000)}
        if key=='1407':
            values = {False:[],True:[]}
            for rep in range(repeats):
                for prune in (bool(rep%2),not bool(rep%2)):
                    start = perf_counter()
                    refine(x,*loc,*args,index,prune)
                    values[prune].append(1000*(perf_counter()-start))
            # The two calls above use the same face solver and the same edge records.
            delta = np.array(values[False][1:])-np.array(values[True][1:])
            if len(delta):
                report['paired_refine_1407_ms'] = {
                    'full':float(np.median(values[False][1:])),'bounded':float(np.median(values[True][1:])),
                    'paired_saving_median':float(np.median(delta)),
                    'paired_saving_p10_p90':np.percentile(delta,[10,90]).tolist()}
    order = np.random.default_rng(825).permutation(len(blocks['1407']))
    ids = np.r_[np.flatnonzero(order<768)[:48],np.flatnonzero(order>=768)[:48]]
    times = np.array([float(row['time_s']) for row in rows])
    frames = [blocks['contact'][times==t] for t in np.unique(times)]
    fixed = blocks['contact'][np.isclose(times,.00022,atol=1e-12,rtol=0.)]
    for name, values in timed(methods,frames,fixed,blocks['1407'][order],ids,repeats).items():
        report['methods'][name].update(values)
    # Record how chord flattening affects distance derivatives away from CAD singularities.
    gradients = np.genfromtxt(folder.parent/'truth-5119.csv',delimiter=',',skip_header=1)[:,1:]
    probes = np.arange(0,len(gradients),12)
    probes = probes[np.isfinite(gradients[probes]).all(axis=1)]
    h = 1e-5
    delta = np.concatenate((np.eye(3),-np.eye(3)))*h
    xyz = (blocks['5119'][probes]+delta[:,None,:]).reshape(-1,3)
    for name,predict,_ in (methods[0],methods[2]):
        v = predict(xyz).reshape(6,-1)
        g = ((v[:3]-v[3:])/(2*h)).T
        cosines = np.einsum('ij,ij->i',g,gradients[probes])/np.linalg.norm(g,axis=1)
        angles = np.degrees(np.arccos(np.clip(cosines,-1,1)))
        report['methods'][name]['normal_check'] = {'n':len(angles),'p95_deg':float(np.percentile(angles,95)),
                                                    'max_deg':float(max(angles))}
    return report


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    mesh = commands.add_parser('mesh')
    mesh.add_argument('output',type=Path)
    mesh.add_argument('--deflection-um',type=float,required=True)
    mesh.add_argument('--step',type=Path,default=STEP)
    mesh.add_argument('--interior',action='store_true')
    mesh.add_argument('--parallel-mesh',action='store_true')
    split = commands.add_parser('split')
    split.add_argument('mesh',type=Path)
    split.add_argument('output',type=Path)
    split.add_argument('--max-edge-mm',type=float,required=True)
    run = commands.add_parser('compare')
    run.add_argument('folder',type=Path)
    run.add_argument('report',type=Path)
    run.add_argument('--repeats',type=int,default=30)
    audit = commands.add_parser('audit')
    audit.add_argument('report',type=Path)
    audit.add_argument('meshes',type=Path,nargs='+')
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.command=='compare':
            if args.repeats<=0:
                parser.error('--repeats must be positive')
            report = compare(args.folder,args.repeats)
            args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            for name,stats in report['methods'].items():
                print(name,stats['mixed_batch_ms'],stats['four_points_us'],flush=True)
        elif args.command=='audit':
            blocks,targets,rows = samples()
            report = {}
            for path in args.meshes:
                with np.load(path) as archive:
                    v,f = archive['v'],archive['f']
                report[path.stem] = {'vertices':len(v),'triangles':len(f),'volume_mm3':check_closed(v,f),
                                     'open3d':quality(mesh_queries(v,f,False),blocks,targets,rows)}
            args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        else:
            if args.command=='mesh':
                if args.deflection_um<=0:
                    parser.error('--deflection-um must be positive')
                v,f = mesh_step(args.deflection_um,args.interior,args.parallel_mesh,args.step)
            else:
                with np.load(args.mesh) as archive:
                    v,f = shorten_edges(archive['v'],archive['f'],args.max_edge_mm)
            volume = check_closed(v,f)
            np.savez(args.output,v=v,f=f)
            print(len(v),len(f),volume)
