"""CPU BVH/HHT foundation runs. Each run advances its own scene from the initial state."""
import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
from time import perf_counter

import numpy as np

from cpu_backend_run import digest
from literature_validation import oscillator_reference

ROOT = Path(__file__).resolve().parents[2]

def aligned_cam(model, directory, segments):
    """Mesh the documented eccentric cylinder with matching angular divisions at every axial layer."""
    import gmsh
    import meshio
    import shutil
    import trimesh

    if segments<8 or segments%4:
        raise ValueError("angular divisions must be a multiple of four")
    directory.mkdir(parents=True, exist_ok=True)
    radius, thickness, offset = .105, .06, .024
    arc_length = 2*np.pi*radius/segments
    gmsh.initialize()
    path = directory/'cam.msh'
    try:
        gmsh.option.setNumber("General.Terminal",0)
        gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
        gmsh.model.add("eccentric cam")
        geo = gmsh.model.geo
        center = geo.addPoint(offset,0,-thickness/2,arc_length)
        corners = [geo.addPoint(offset+radius*x,radius*y,-thickness/2,arc_length)
                   for x,y in [(1,0),(0,1),(-1,0),(0,-1)]]
        edges = [geo.addCircleArc(corners[i],center,corners[(i+1)%4]) for i in range(4)]
        disk = geo.addPlaneSurface([geo.addCurveLoop(edges)])
        extrusion = geo.extrude([(2,disk)],0,0,thickness,numElements=[max(1,round(thickness/arc_length))])
        geo.synchronize()
        for edge in edges:
            gmsh.model.mesh.setTransfiniteCurve(edge,segments//4+1)
        volume = next(entity for dim,entity in extrusion if dim==3)
        exterior = gmsh.model.getBoundary([(3,volume)],oriented=False)
        gmsh.model.addPhysicalGroup(2,[tag for _,tag in exterior])
        gmsh.model.mesh.generate(3)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    mesh = meshio.read(path)
    tri = trimesh.Trimesh(mesh.points,mesh.get_cells_type('triangle'),process=True)
    if not tri.is_watertight or tri.body_count!=1:
        raise ValueError("cam boundary is not one closed surface")
    tri.fix_normals()
    if not tri.is_volume:
        raise ValueError("cam normals do not orient a positive volume")
    side=tri.triangles[np.abs(tri.face_normals[:,2])<.5]
    radial=side[:,:,:2]-[offset,0]
    if not np.allclose(np.linalg.norm(radial,axis=2),radius,rtol=0,atol=1e-10):
        raise ValueError("cam side nodes depart from the circular CAD surface")
    midpoints=.5*(radial+np.roll(radial,1,axis=1))
    chord=float(radius-np.linalg.norm(midpoints,axis=2).min())
    if abs(chord-radius*(1-np.cos(np.pi/segments)))>1e-9:
        raise ValueError("axial rings do not have the specified common angular divisions")
    meshio.write_points_cells(path,tri.vertices,[("triangle",tri.faces)],file_format='gmsh22',binary=False,
                              cell_data={tag:[np.ones(len(tri.faces),int)] for tag in ['gmsh:physical','gmsh:geometrical']})
    lengths=np.linalg.norm(tri.triangles-np.roll(tri.triangles,-1,axis=1),axis=2)
    quality=4*np.sqrt(3)*tri.area_faces/(lengths**2).sum(axis=1)
    info=dict(triangles=len(tri.faces),vertices=len(tri.vertices),axial_alignment=True,segments=segments,
              surface_chord_error_m=chord,min_triangle_mean_ratio=float(quality.min()))
    (directory/'cam.quality.json').write_text(json.dumps(info,indent=2)+'\n')
    for name in ['follower.msh','follower.quality.json']:
        shutil.copyfile(model/'compact-planar'/name,directory/name)


def box(path, origin, size):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.add(path.stem)
        gmsh.model.occ.addBox(*origin, *size)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path.with_suffix(".step")))
        gmsh.option.setNumber("Mesh.MeshSizeMin", max(size))
        gmsh.option.setNumber("Mesh.MeshSizeMax", max(size))
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def prepare(work):
    work.mkdir(parents=True, exist_ok=True)
    box(work/"floor.msh", (-.05,-.05,-.02), (.10,.10,.02))
    box(work/"slider.msh", (-.01,-.01,0), (.02,.02,.02))
    fixed='body "floor.step" 0 0 0 0 0 0 1 0 0 0 0 -1 0 0 0 0'
    slider='body "slider.step" 0 0 0 0 0 0 1 0 0 0 0 -1 0 0 0 1 0 0 0 -1'
    osc=['nc-contact-v2','gravity 0 0 0 1 2 1','coordinate 1 .001 0 1 0 0 0',
         fixed, slider,'surface 0 1 .1 1 0 0 .0005 .002','drives 0']
    (work/'oscillator.scene').write_text('\n'.join(osc)+'\n')
    # The contact faces are 18 mm apart. Relative and center-of-mass motion have separate exact solutions.
    free_a='body "slider.step" 0 0 0 -.02 0 0 1 0 0 0 0 -1 0 0 0 1 0 0 0 1'
    free_b='body "slider.step" 0 0 0 .018 0 0 1 0 0 0 0 -1 0 0 0 1 1 0 0 1'
    impact=['nc-contact-v2','gravity 0 0 0 2 2 1','coordinate 1 0 0 0 0 0 0','coordinate 1 0 -.45 0 0 0 0',
            free_a, free_b,'surface 0 1 5000 1 0 0 .0005 .008','drives 0']
    (work/'impact.scene').write_text('\n'.join(impact)+'\n')


def run(exe, work, case, method, step, directory, cam_mesh="compact-planar", cam_root=None):
    directory.mkdir(parents=True, exist_ok=False)
    scene = work/(case+'.scene')
    args = []
    duration = {'impact':5., 'oscillator':20., 'cam':5.}[case]
    if case=='cam':
        cam_root = cam_root or ROOT/'build/cpu-contact/cam'
        scene=cam_root/'model/cam.scene'
        args=['--mesh-dir', cam_root/cam_mesh]
    cmd=[exe,scene,'0',str(step),str(duration),'1',directory/'response','--method',method,
         '--solver','penalty','--mesh-region-scope','finite','--progress',directory/'progress.csv',*args]
    report=dict(case=case,method=method,scope='finite',solver='HHT penalty',step_s=step,requested_s=duration,
                exe_sha256=digest(exe),scene_sha256=digest(scene))
    asset_dir = Path(args[1]) if args else scene.parent
    meshes = {shlex.split(line)[1] for line in scene.read_text().splitlines() if line.startswith("body ")}
    report["mesh_sha256"] = {name: digest((asset_dir/name).with_suffix(".msh")) for name in sorted(meshes)}
    start=perf_counter()
    try:
        r=subprocess.run(cmd,text=True,capture_output=True,timeout=1800)
        report.update(status='complete' if r.returncode==0 else 'failed',exit_code=r.returncode,message=(r.stderr or r.stdout).strip())
    except subprocess.TimeoutExpired:
        report.update(status='timeout')
    report['wall_s']=perf_counter()-start
    if report['status']=='complete':
        d=np.genfromtxt(directory/'response.csv',names=True,delimiter=',')
        t=d['time_s']
        np.testing.assert_allclose(t,np.arange(round(duration/step)+1)*step,atol=1e-9,rtol=0)
        for c in d.dtype.names:
            if c.endswith('_gap_m'):
                active=d[c[:-6]+'_hit'].astype(bool)
                if np.isinf(d[c]).any() or not np.isfinite(d[c][active]).all():
                    raise ValueError(f"nonfinite active gap in {c}")
            elif not np.isfinite(d[c]).all():
                raise ValueError(f"nonfinite response in {c}")
        f=d['p0_normal_N']
        if case=='oscillator':
            q,contact_quarter,free_duration,period=oscillator_reference(t,1.,.1,.001)
            expected=.1*np.maximum(q,0)
            indices=np.flatnonzero(np.diff(np.signbit(d['q0'])))
            actual=t[indices]-step*d['q0'][indices]/(d['q0'][indices+1]-d['q0'][indices])
            events=np.sort(np.r_[np.arange(contact_quarter,duration,period),np.arange(contact_quarter+free_duration,duration,period)])
            report['transitions']=len(actual)
            report['event_error_s']=float(np.max(abs(actual-events))) if len(events)==len(actual) else None
            report['q_max_error_m']=float(np.max(abs(d['q0']-q)))
            report['force_max_error_N']=float(np.max(abs(f-expected)))
        if case=='impact':
            entry=.018/.45
            omega=np.sqrt(5000./.5)
            release=entry+np.pi/omega
            tau=np.clip(t-entry,0,release-entry)
            expected=5000*.45/omega*np.sin(omega*tau)
            expected[(t<entry)|(t>release)]=0
            relative=np.where(t<entry,.018-.45*t,np.where(t<release,-.45/omega*np.sin(omega*tau),.45*(t-release)))
            report.update(gap_max_error_m=float(np.max(abs(.018+d['q1']-d['q0']-relative))),
                          force_max_error_N=float(np.max(abs(f-expected))),
                          final_v_m_s=[float(d['v0'][-1]),float(d['v1'][-1])],
                          momentum_max_error=float(np.max(abs(d['v0']+d['v1']+.45))),
                          peak_force_N=float(max(f)), exact_peak_force_N=5000*.45/omega,
                          switches=t[np.flatnonzero(np.diff(f>0))+1].tolist(), exact_events_s=[entry,release])
        with (directory/'response.stats.csv').open() as stream: report['stats']=next(csv.DictReader(stream))
    (directory/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(case,method,report['status'],report['wall_s'],report.get('message',''), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exe',type=Path)
    parser.add_argument('work',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--cases',nargs='+',default=['impact','oscillator','cam'])
    parser.add_argument('--methods',nargs='+',default=['mesh-aabb','mesh-obb'])
    parser.add_argument('--step',type=float,default=.0001)
    parser.add_argument('--cam-mesh',default='compact-planar')
    parser.add_argument('--cam-root',type=Path)
    a=parser.parse_args()
    if a.prepare:
        # Gmsh loads its own OCCT DLLs on Windows. Keep its loader state outside the simulator's parent process.
        import sys
        subprocess.run([sys.executable,'-c','from pathlib import Path; from cpu_foundation import prepare; import sys; prepare(Path(sys.argv[1]))',str(a.work.resolve())],cwd=Path(__file__).resolve().parent,check=True)
    for case in a.cases:
        for method in a.methods:
            run(a.exe.resolve(),a.work.resolve(),case,method,a.step,(a.out/(case+'-'+method)).resolve(),a.cam_mesh,a.cam_root)
