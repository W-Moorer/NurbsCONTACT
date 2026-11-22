"""Double-precision closest-triangle BVH for a STEP face or a complete rigid mesh."""
from hashlib import sha256
from time import perf_counter

import cupy as cp
import numpy as np
from scipy.spatial import cKDTree

from nhrep_gear import read_occt_solid


TRIANGLE_CUDA = r'''
// Closest point on a triangle by its vertex, edge and face Voronoi regions.
__device__ double3 sub3(double3 a,double3 b) {return make_double3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ double prod(double3 a,double3 b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ double3 step3(double3 a,double s,double3 b) {
    return make_double3(a.x+s*b.x,a.y+s*b.y,a.z+s*b.z);
}
__device__ double3 foot(double3 p,const double3* t) {
    double3 a=t[0], b=t[1], c=t[2], ab=sub3(b,a), ac=sub3(c,a);
    double3 ap=sub3(p,a); double d1=prod(ab,ap), d2=prod(ac,ap);
    if (d1<=0 && d2<=0) return a;
    double3 bp=sub3(p,b); double d3=prod(ab,bp), d4=prod(ac,bp);
    if (d3>=0 && d4<=d3) return b;
    double vc=d1*d4-d3*d2;
    if (vc<=0 && d1>=0 && d3<=0) return step3(a,d1/(d1-d3),ab);
    double3 cp=sub3(p,c); double d5=prod(ab,cp), d6=prod(ac,cp);
    if (d6>=0 && d5<=d6) return c;
    double vb=d5*d2-d1*d6;
    if (vb<=0 && d2>=0 && d6<=0) return step3(a,d2/(d2-d6),ac);
    double va=d3*d6-d5*d4;
    if (va<=0 && d4>=d3 && d5>=d6)
        return step3(b,(d4-d3)/((d4-d3)+(d5-d6)),sub3(c,b));
    double inv=1.0/(va+vb+vc);
    return step3(step3(a,vb*inv,ab),vc*inv,ac);
}
__device__ double distance2(double3 p,const double3* t) {
    double3 d=sub3(p,foot(p,t));
    return prod(d,d);
}
__device__ double box_distance(const double* boxes,int node,const double* p) {
    double sum=0;
    for(int k=0;k<3;++k) {
        double d=fmax(0.,fmax(boxes[6*node+k]-p[k],p[k]-boxes[6*node+3+k]));
        sum+=d*d;
    }
    return sum;
}
'''

CUDA = TRIANGLE_CUDA + r'''
extern "C" __global__ void mesh_line(const int* range,const int* offsets,const int* roots,
    const int4* nodes,const double* boxes,const double3* tri,
    const double* samples,const double* centers,const int* prior,double cutoff,double stiffness,
    int* ids,double* witnesses,double* wrenches) {
    int line=range[0]+blockIdx.x;
    if (line>=range[1]) return;
    int warp=threadIdx.x/32, lane=threadIdx.x%32, root=roots[line];
    double w[6]={};
    for(int j=offsets[line]+warp;j<offsets[line+1];j+=blockDim.x/32) {
        const double* sample=samples+4*j;
        double3 p=make_double3(sample[0],sample[1],sample[2]);
        int best=prior[j]>=0 ? ids[prior[j]] : -1;
        if(best<0) {
            int t=root;
            while(nodes[t].y<0) {
                int left=t+1,right=nodes[t].w;
                t=box_distance(boxes,left,sample)<=box_distance(boxes,right,sample) ? left:right;
            }
            best=nodes[t].y;
        }
        double sq=distance2(p,tri+3*best);
        for(int t=root;t<nodes[root].x;) {
            int4 node=nodes[t];
            if(box_distance(boxes,t,sample)>sq) {t=node.x; continue;}
            ++t;
            if(node.y<0) continue;
            for(int begin=node.y;begin<node.z;begin+=32) {
                int id=begin+lane;
                double d=id<node.z ? distance2(p,tri+3*id) : 1.7976931348623157e308;
                for(int shift=16;shift;shift/=2) {
                    double other=__shfl_down_sync(0xffffffff,d,shift);
                    int index=__shfl_down_sync(0xffffffff,id,shift);
                    if(other<d || (other==d && index<id)) { d=other; id=index; }
                }
                d=__shfl_sync(0xffffffff,d,0); id=__shfl_sync(0xffffffff,id,0);
                if(d<sq || (d==sq && id<best)) {sq=d;best=id;}
            }
        }
        if(lane==0) {
            ids[j]=best;
            if(best<0) {for(int k=0;k<6;++k) w[k]=nan(""); continue;}
            double3 x=foot(p,tri+3*best);
            witnesses[3*j]=x.x; witnesses[3*j+1]=x.y; witnesses[3*j+2]=x.z;
            double d[3]={p.x-x.x,p.y-x.y,p.z-x.z};
            add_point_wrench(sample,centers+3*line,d,sqrt(sq),cutoff,stiffness,w);
        }
    }
    __shared__ double partial[4][6];
    if(lane==0) for(int k=0;k<6;++k) partial[warp][k]=w[k];
    __syncthreads();
    if(threadIdx.x<6) {
        double total=0;
        for(int s=0;s<4;++s) total+=partial[s][threadIdx.x];
        wrenches[6*line+threadIdx.x]=total;
    }
}
'''


def triangulate(step, deflection_um, face_ids):
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IMeshTools import IMeshTools_Parameters
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS

    solid = read_occt_solid(step)
    settings = IMeshTools_Parameters()
    settings.Deflection, settings.Angle = deflection_um/1000, .1
    settings.Relative = settings.ControlSurfaceDeflection = settings.InParallel = False
    faces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(solid, TopAbs_FACE, faces)
    for i in face_ids:
        face = TopoDS.Face_s(faces.FindKey(i))
        if not BRepMesh_IncrementalMesh(face, settings).IsDone():
            raise ValueError(f'STEP tessellation failed on face {i}')
        loc = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face, loc)
        tr = loc.Transformation()
        xyz = np.array([mesh.Node(j).Transformed(tr).Coord() for j in range(1,mesh.NbNodes()+1)])
        idx = np.array([t.Get() for t in mesh.Triangles()])-1
        triangles = xyz[idx]
        if not len(idx) or np.any(np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                                                          triangles[:,2]-triangles[:,0]),axis=1)==0):
            raise ValueError(f'Empty face or degenerate triangle on face {i}')
        yield i, triangles


def cpu_foot(p, tri):
    """Exhaustive face scan using plane projections and three clamped segments."""
    a, b, c = tri.transpose(1,0,2)
    n = np.cross(b-a,c-a)
    sq = np.einsum('ij,ij->i',n,n)
    projected = p - (np.einsum('ij,ij->i',p-a,n)/sq)[:,None]*n
    inside = np.ones(len(a),bool)
    candidates = []
    for u,v in ((a,b),(b,c),(c,a)):
        edge = v-u
        inside &= np.einsum('ij,ij->i',np.cross(edge,projected-u),n)>=0
        s = np.einsum('ij,ij->i',p-u,edge)/np.einsum('ij,ij->i',edge,edge)
        candidates.append(u+np.clip(s,0,1)[:,None]*edge)
    candidates.append(projected[inside])
    all_points = np.concatenate(candidates)
    return all_points[np.argmin(np.linalg.norm(p-all_points,axis=1))]


def read_full_mesh(full_mesh):
    with np.load(full_mesh) as m:
        vertices, faces = np.asarray(m['v'], dtype=np.float64), m['f']
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or not np.issubdtype(faces.dtype,np.integer) or faces.min()<0 or faces.max()>=len(vertices)):
            raise ValueError(f'Invalid triangle coordinates or indices in {full_mesh}')
        triangles = vertices[faces]
        if np.any(np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                                          triangles[:,2]-triangles[:,0]),axis=1)==0):
            raise ValueError(f'Degenerate triangle in {full_mesh}')
        return triangles


def device_bvhs(faces):
    ordered, bounds, nodes, roots = [], [], [], {}
    total = 0
    for face_id, tri in faces.items():
        tree = cKDTree(tri.mean(axis=1),leafsize=32)
        block = tri[tree.indices]
        ordered.append(block)

        def visit(node):
            i = len(nodes)
            nodes.append(None)
            bounds.append(None)
            if node.split_dim < 0:
                a, b = node.start_idx, node.end_idx
                bounds[i] = [block[a:b].min(axis=(0,1)), block[a:b].max(axis=(0,1))]
                nodes[i] = [i+1,total+a,total+b,-1]
            else:
                left, right = visit(node.lesser), visit(node.greater)
                bounds[i] = [np.minimum(bounds[left][0],bounds[right][0]),
                             np.maximum(bounds[left][1],bounds[right][1])]
                nodes[i] = [len(nodes),-1,-1,right]
            return i

        roots[face_id] = visit(tree.tree)
        total += len(block)
    return roots, (cp.asarray(nodes,dtype=cp.int32), cp.asarray(bounds,dtype=cp.float64),
                   cp.asarray(np.concatenate(ordered)))


class MeshLines:
    def __init__(self, step, deflection, keys, max_lines, offsets, points, centers, prior, cutoff, stiffness, common,
                 full_mesh=None):
        start = perf_counter()
        self.face_ids = [k[1] for k in keys]
        self.source = dict(scope='independently tessellated supplied STEP faces',deflection_um=deflection)
        if full_mesh is None:
            self.faces = dict(triangulate(step, deflection, sorted(set(self.face_ids))))
        else:
            self.faces = {0: read_full_mesh(full_mesh)}
            self.face_ids = [0]*len(keys)
            self.source = dict(scope='whole-body closest triangle',mesh_file=full_mesh.name,
                               sha256=sha256(full_mesh.read_bytes()).hexdigest())
        roots, geometry = device_bvhs(self.faces)
        self.keys, self.offsets, self.points = keys, offsets, np.asarray(points)
        self.max_lines = max_lines
        self.kernel = cp.RawKernel(common+CUDA,'mesh_line')
        line_roots = [roots[i] for i in self.face_ids]
        self.inputs = tuple(cp.asarray(x,dtype=cp.int32) for x in (offsets,line_roots)) + geometry
        self.inputs += (cp.asarray(self.points),cp.asarray(centers),cp.asarray(prior),
                        np.float64(cutoff),np.float64(stiffness))
        self.ids = cp.full(len(points),-1,cp.int32)
        self.witness = cp.full((len(points),3),np.nan)
        self.wrench = cp.full((len(keys),6),np.nan)
        cp.cuda.get_current_stream().synchronize()
        self.setup_s = perf_counter()-start

    def launch(self, interval):
        self.kernel((self.max_lines,), (128,), (interval,*self.inputs,self.ids,self.witness,self.wrench))

    def check(self, ranges, target_gaps, target_wrenches):
        for r in ranges:
            self.launch(cp.asarray(r))
        feet, w = self.witness.get(), self.wrench.get()
        if not np.isfinite(w).all() or np.any(self.ids.get()<0):
            raise ValueError('Incomplete mesh wrench')
        ref = np.array([cpu_foot(p[:3],self.faces[face_id]) for i,face_id in enumerate(self.face_ids)
                        for p in self.points[self.offsets[i]:self.offsets[i+1]]])
        np.testing.assert_allclose(feet, ref, atol=1e-8, rtol=0)
        gaps = np.linalg.norm(self.points[:,:3]-feet,axis=1)
        err = w-np.array(target_wrenches)
        return dict(**self.source,triangles=sum(map(len,self.faces.values())),
                    prepare_bvh_and_upload_s=self.setup_s, max_gap_error_um=float(1000*max(abs(gaps-target_gaps))),
                    max_line_force_vector_error_N=float(max(np.linalg.norm(err[:,:3],axis=1))),
                    max_line_moment_vector_error_Nm=float(max(np.linalg.norm(err[:,3:],axis=1))))
