"""Closed oriented triangle mesh baseline with double precision and feature pseudonormals."""
import igl
import numpy as np
from numba import njit

from fitted_locator import nearby, pack_grid, pack_tree


@njit(cache=True)
def evaluate(points, vertices, triangles, grads, face_n, vertex_n, edge_n, tree, grid):
    stack = np.empty(tree[3],np.int32)
    result = np.empty(len(points))
    for i,p in enumerate(points):
        sq,t,x,y,z = nearby(p,np.inf,vertices,triangles,grads,tree,grid,stack)
        a = vertices[triangles[t,0]]
        dx,dy,dz = x-a[0],y-a[1],z-a[2]
        v = dx*grads[t,0,0]+dy*grads[t,0,1]+dz*grads[t,0,2]
        w = dx*grads[t,1,0]+dy*grads[t,1,1]+dz*grads[t,1,2]
        bary = (1-v-w,v,w)
        normal = face_n[t]
        support, corner, opposite = 0, 0, 0
        for k in range(3):
            if bary[k]>1e-9:
                support += 1
                corner = k
            else:
                opposite = k
        if support==1:
            normal = vertex_n[triangles[t,corner]]
        elif support==2:
            normal = edge_n[t,opposite]
        direction = (p[0]-x)*normal[0]+(p[1]-y)*normal[1]+(p[2]-z)*normal[2]
        result[i] = np.copysign(np.sqrt(sq),direction)
    return result


def mesh_query(vertices, triangles):
    triangles = np.ascontiguousarray(triangles,dtype=np.int64)
    a,b,c = vertices[triangles].transpose(1,0,2)
    n = np.cross(b-a,c-a)
    sq = np.einsum('ij,ij->i',n,n)
    if np.any(sq==0):
        raise ValueError('Degenerate guide triangle')
    face_n = n/np.sqrt(sq)[:,None]
    grads = np.stack((np.cross(c-a,n),np.cross(n,b-a)),axis=1)/sq[:,None,None]
    vertex_n = igl.per_vertex_normals(vertices,triangles,igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE)
    start,end = np.roll(triangles,-1,axis=1),np.roll(triangles,-2,axis=1)
    keys = np.minimum(start,end)*len(vertices)+np.maximum(start,end)
    _,inverse,counts = np.unique(keys.ravel(),return_inverse=True,return_counts=True)
    if np.any(counts!=2):
        raise ValueError('Pseudonormals require a closed manifold mesh')
    edge_n = np.zeros((len(counts),3))
    np.add.at(edge_n,inverse,np.repeat(face_n,3,axis=0))
    edge_n = edge_n[inverse].reshape(len(triangles),3,3)
    tree = pack_tree(vertices,triangles)
    grid = pack_grid(vertices,tree,min(128,max(1,round(2*np.cbrt(len(triangles))))))

    def predict(points):
        points = np.asarray(points,dtype=np.float64)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
            raise ValueError('Expected finite XYZ coordinates in millimetres')
        return evaluate(points,vertices,triangles,grads,face_n,vertex_n,edge_n,tree,grid)
    return predict
