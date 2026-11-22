"""Single-thread compiled kernels for projection onto fitted Chebyshev faces."""
import numpy as np
from numba import njit
from trim_bounds import edge_minimum


def pack_faces(faces):
    size = max(faces) + 1
    offsets = np.zeros((size, 3), np.int64)
    frames = np.zeros((size, 5))
    blocks = []
    start = 0
    for face, (center, half, coef, _, _, sign) in faces.items():
        nu, nv, _ = coef.shape
        frames[face] = (*center, *half, sign)
        offsets[face] = (start, nu, nv)
        blocks.append(coef.reshape(-1, 3))
        start += nu*nv
    return np.ascontiguousarray(np.concatenate(blocks)), offsets, frames


@njit(cache=True)
def weights_at(point, triangle, xyz, tri, bary_grad):
    v, w = 0., 0.
    for k in range(3):
        delta = point[k]-xyz[tri[triangle,0],k]
        v += delta*bary_grad[triangle,0,k]
        w += delta*bary_grad[triangle,1,k]
    return np.array((1-v-w, v, w))


@njit(cache=True, inline='always')
def dot3(a,b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


@njit(cache=True)
def fill_basis(x, count, buf):
    buf[0,0],buf[0,1] = 1.,0.
    if count>1:
        buf[1,0],buf[1,1] = x,1.
    for i in range(2,count):
        buf[i,0] = 2*x*buf[i-1,0]-buf[i-2,0]
        buf[i,1] = 2*(buf[i-1,0]+x*buf[i-1,1])-buf[i-2,1]


@njit(cache=True)
def evaluate_into(chart,face,coefficients,offsets,bu,bv,out):
    start,nu,nv = offsets[face]
    fill_basis(chart[0],nu,bu)
    fill_basis(chart[1],nv,bv)
    if nv == 2:
        # Degree one in v: evaluate and differentiate that row directly.
        v = chart[1]
        for k in range(3):
            value, du, dv = 0., 0., 0.
            for i in range(nu):
                slope = coefficients[start+2*i+1,k]
                row = coefficients[start+2*i,k] + v*slope
                value += bu[i,0]*row
                du += bu[i,1]*row
                dv += bu[i,0]*slope
            out[0,k], out[1,k], out[2,k] = value, du, dv
        return
    for k in range(3):
        value, du, dv = 0., 0., 0.
        for i in range(nu):
            row, row_v = 0., 0.
            for j in range(nv):
                c = coefficients[start+i*nv+j,k]
                row += bv[j,0]*c
                row_v += bv[j,1]*c
            value += bu[i,0]*row
            du += bu[i,1]*row
            dv += bu[i,0]*row_v
        out[0,k], out[1,k], out[2,k] = value, du, dv


@njit(cache=True)
def chart_normal(target,chart,face,coefficients,offsets,frames,bu,bv,out,normal):
    for it in range(3):
        evaluate_into(chart,face,coefficients,offsets,bu,bv,out)
        p,u,v = out[0],out[1],out[2]
        if it==2:
            break
        a,b,c = dot3(u,u),dot3(u,v),dot3(v,v)
        det = a*c-b*b
        if det<=1e-10*(a+c)**2:
            change = np.linalg.pinv(out[1:].T) @ (target-p)
            du,dv = change[0],change[1]
        else:
            ru,rv = 0.,0.
            for k in range(3):
                ru += u[k]*(target[k]-p[k])
                rv += v[k]*(target[k]-p[k])
            du,dv = (c*ru-b*rv)/det,(a*rv-b*ru)/det
        if a*du*du+2*b*du*dv+c*dv*dv < 1e-26:
            break
        chart[0] += du
        chart[1] += dv
    u,v = out[1],out[2]
    normal[0] = u[1]*v[2]-u[2]*v[1]
    normal[1] = u[2]*v[0]-u[0]*v[2]
    normal[2] = u[0]*v[1]-u[1]*v[0]
    scale = frames[face,4]/np.sqrt(dot3(normal,normal))
    for k in range(3):
        normal[k] *= scale


@njit(cache=True)
def project_one(x, seed, t, xyz, tri, owner, uv, bary_grad, coefficients, offsets, frames, bu,bv,out):
    face = owner[t]
    weights = weights_at(seed,t,xyz,tri,bary_grad)
    chart = np.empty(2)
    for j in range(2):
        at_seed = weights[0]*uv[t,0,j]+weights[1]*uv[t,1,j]+weights[2]*uv[t,2,j]
        chart[j] = (at_seed-frames[face,j])/frames[face,2+j]
    current = seed.copy()
    normal = np.empty(3)
    for _ in range(4):
        chart_normal(current,chart,face,coefficients,offsets,frames,bu,bv,out,normal)
        distance = 0.
        for k in range(3):
            distance += (x[k]-out[0,k])*normal[k]
        movement = 0.
        for k in range(3):
            value = x[k]-distance*normal[k]
            movement += (current[k]-value)**2
            current[k] = value
        if movement < 1e-26:
            break
    return current, normal


@njit(cache=True)
def projected(x, seed, triangles, xyz, tri, owner, uv, bary_grad, coefficients, offsets, frames, bu,bv,out):
    count = len(triangles)
    if count==1:
        return project_one(x,seed,triangles[0],xyz,tri,owner,uv,bary_grad,coefficients,offsets,frames,bu,bv,out)
    charts = np.empty((count,2))
    for i, t in enumerate(triangles):
        weights = weights_at(seed,t,xyz,tri,bary_grad)
        frame = frames[owner[t]]
        for j in range(2):
            at_seed = weights[0]*uv[t,0,j]+weights[1]*uv[t,1,j]+weights[2]*uv[t,2,j]
            charts[i,j] = (at_seed-frame[j])/frame[2+j]
    current = seed.copy()
    normals = np.empty((count,3))
    distances = np.empty(count)
    correction = np.empty(3)
    for _ in range(4):
        for i,t in enumerate(triangles):
            chart_normal(current,charts[i],owner[t],coefficients,offsets,frames,bu,bv,out,normals[i])
            distances[i] = 0.
            for k in range(3):
                distances[i] += (x[k]-out[0,k])*normals[i,k]
        if count==2:
            # Retain the same relative eigenvalue threshold as the reference SVD.
            dot = dot3(normals[0],normals[1])
            r,s = distances
            if 1-abs(dot)>1e-4*(1+abs(dot)):
                a,b = (r-dot*s)/(1-dot*dot),(s-dot*r)/(1-dot*dot)
            else:
                sign = 1. if dot>=0 else -1.
                a = (r+sign*s)/(2*(1+abs(dot)))
                b = sign*a
            for k in range(3):
                correction[k] = a*normals[0,k]+b*normals[1,k]
        else:
            gram = normals.T @ normals
            correction[:] = np.linalg.pinv(gram,rcond=1e-4) @ (normals.T @ distances)
        movement = 0.
        for k in range(3):
            value = x[k]-correction[k]
            movement += (value-current[k])**2
            current[k] = value
        if movement < 1e-26:
            break
    return current, normals.sum(axis=0)


@njit(cache=True)
def supports(weights, nearest, tri, owner, incident):
    pos = np.nonzero(weights>1e-9)[0]
    if len(pos) != 1 and len(pos) != 2:
        return np.array([nearest])
    shared = tri[nearest,pos]
    nearby = incident[shared[0]]
    chosen = np.empty(len(nearby),np.int64)
    count = 0
    for t in nearby:
        if t < 0:
            continue
        if len(shared)==2 and not np.any(tri[t]==shared[1]):
            continue
        duplicate = False
        for k in range(count):
            if owner[t] == owner[chosen[k]]:
                duplicate = True
                # Reference support uses the first incident triangle, or
                # ascending triangle indices at a shared mesh edge.
                if len(shared)==2 and t<chosen[k]:
                    chosen[k] = t
                break
        if not duplicate:
            chosen[count] = t
            count += 1
    return chosen[:count]


@njit(cache=True, inline='always')
def any_incident(triangles, owner, first, second):
    for t in triangles:
        if owner[t]==first or owner[t]==second:
            return True
    return False


@njit(cache=True)
def refine(points, nearest, seeds, edge_sq, edge, edge_seeds, xyz, tri, owner, uv, incident,
           edge_pairs, convex_sign, bary_grad, coefficients, offsets, frames, crease_band_mm,
           trim_index=None, prune=True, counts=None):
    bu = np.empty((np.max(offsets[:,1]),2))
    bv = np.empty((np.max(offsets[:,2]),2))
    out = np.empty((3,3))
    result = np.empty(len(points))
    stack = np.empty(1 if trim_index is None else trim_index[4],np.int32)
    for k in range(len(points)):
        x, t, seed = points[k], nearest[k], seeds[k]
        weights = weights_at(seed,t,xyz,tri,bary_grad)
        gap_sq = 0.
        for j in range(3):
            gap_sq += (x[j]-seed[j])**2
        near = (len(edge)>0 and gap_sq <= crease_band_mm**2
                and edge_sq[k] <= (2*np.sqrt(gap_sq)+.005)**2)
        if not near and min(weights[0],weights[1],weights[2])>1e-9:
            p,n = project_one(x,seed,t,xyz,tri,owner,uv,bary_grad,coefficients,offsets,frames,bu,bv,out)
            delta = x-p
            result[k] = np.copysign(np.sqrt(dot3(delta,delta)),dot3(delta,n))
            continue
        local = supports(weights,t,tri,owner,incident)
        if near:
            a,b = edge_pairs[edge[k]]
            near = len(local)<=2 and any_incident(local, owner, owner[a], owner[b])
        if not near:
            p,n = projected(x,seed,local,xyz,tri,owner,uv,bary_grad,coefficients,offsets,frames,bu,bv,out)
            result[k] = np.copysign(np.linalg.norm(x-p), (x-p)@n)
            continue
        # In the crease region the wedge calculation replaces the initial
        # projection, so the initial projection can be skipped entirely.
        sides, start = edge_pairs[edge[k]], edge_seeds[k]
        sign = convex_sign[edge[k]]
        p,n = project_one(x,start,sides[0],xyz,tri,owner,uv,bary_grad,coefficients,offsets,frames,bu,bv,out)
        q,m = project_one(x,start,sides[1],xyz,tri,owner,uv,bary_grad,coefficients,offsets,frames,bu,bv,out)
        dist = np.inf
        if sign*((p-q)@m)<=0:
            dist = np.linalg.norm(x-p)
        if sign*((q-p)@n)<=0:
            dist = min(dist,np.linalg.norm(x-q))
        if trim_index is not None and trim_index[0][edge[k]]>=0:
            roots,boxes,links,chords,_ = trim_index
            bound = dist if prune else np.inf
            value,tested = edge_minimum(x,roots[edge[k]],boxes,links,chords,stack,bound)
            dist = min(dist,value)
            if counts is not None:
                counts[0] += 1
                counts[1] += int(tested==0)
                counts[2] += tested
        else:
            joint,_ = projected(x,start,sides,xyz,tri,owner,uv,bary_grad,coefficients,offsets,frames,bu,bv,out)
            dist = min(dist,np.linalg.norm(x-joint))
        result[k] = sign*np.copysign(dist,max(sign*((x-p)@n),sign*((x-q)@m)))
    return result
