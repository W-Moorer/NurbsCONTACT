"""Compare the global cube tangent field and locally fitted trimmed-face geometry on STEP."""
import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

import igl
import numpy as np
from numpy.polynomial import chebyshev as cheb
import trimesh
import torch
from threadpoolctl import threadpool_limits

from cube_field import TangentField
from nhrep_gear import cad_triangle_parameters


def probe_set(mesh, geometry, seed, count=768):
    rng = np.random.default_rng(seed)
    vertices = geometry['vertices']
    normals = geometry['normals']
    triangles = rng.choice(len(mesh.faces), count, p=mesh.area_faces / mesh.area_faces.sum())
    bary = rng.dirichlet(np.ones(3), size=count)
    surface = np.einsum('ij,ijk->ik', bary, vertices[mesh.faces[triangles]])
    face = surface + rng.uniform(-.05, .05, (count, 1))*normals[triangles]
    segments = geometry['feature_segments']
    edges = segments[rng.choice(len(segments), count)]
    t = rng.uniform(.05, .95, (count, 1))
    edge = vertices[edges[:, 0]]*(1-t) + vertices[edges[:, 1]]*t
    w = rng.uniform(.05, .95, (count, 1))
    n = w*normals[edges[:, 2]] + (1-w)*normals[edges[:, 3]]
    sharp = edge + rng.uniform(-.05, .05, (count, 1))*n/np.linalg.norm(n, axis=1)[:, None]
    return np.vstack((face, sharp))


def prepare(step, mesh_trial, exe, output):
    meta = json.loads((mesh_trial/'spur.json').read_text())
    mesh = trimesh.load_mesh(mesh_trial/'spur.obj', process=False)
    geometry = np.load(mesh_trial/'spur-geometry.npz')
    owners, uv = cad_triangle_parameters(step, meta['deflection_mm'], mesh,
                                          np.array(meta['center_mm']), meta['mm_per_network_unit'])
    np.savez_compressed(output/'guide.npz', vertices=geometry['vertices'], triangles=mesh.faces.astype(np.int32),
                        owners=owners.astype(np.int16), uv=uv)
    for seed in (801, 903, 1407):
        path = output/f'probe-{seed}.txt'
        np.savetxt(path, probe_set(mesh, geometry, seed), fmt='%.17g')
        subprocess.run([exe.resolve(), 'cad-point-distance', step, path, output/f'truth-{seed}.csv'], check=True)


def basis(uv, degree):
    return cheb.chebvander2d(uv[:, 0], uv[:, 1], degree)


def fit_face(uv, xyz, normals, rng):
    low, high = uv.min(0), uv.max(0)
    center, half = (low+high)/2, (high-low)/2
    scaled = (uv-center)/half
    order = rng.permutation(len(uv))
    split = len(uv)*4//5
    train, valid = order[:split], order[split:]
    choices = [(1, 1), (3, 1), (1, 3), (5, 1), (1, 5), (8, 1), (1, 8), (12, 1), (1, 12),
               (16, 1), (1, 16), (3, 3), (5, 5), (8, 8), (12, 12)]
    best = np.inf
    for degree in choices:
        design = basis(scaled, degree)
        if design.shape[1]*1.5 > len(train):
            continue
        c = np.linalg.lstsq(design[train], xyz[train], rcond=1e-12)[0]
        error = np.max(np.linalg.norm(design[valid]@c-xyz[valid], axis=1))
        if error < best:
            best, chosen = error, degree
        if best < 1e-9:
            break
    design = basis(scaled, chosen)
    c = np.linalg.lstsq(design, xyz, rcond=1e-12)[0].reshape(chosen[0]+1, chosen[1]+1, 3)
    u, v = scaled[0]
    du, dv = cheb.chebder(c, axis=0), cheb.chebder(c, axis=1)
    sign = np.sign(np.dot(np.cross(cheb.chebval2d(u,v,du), cheb.chebval2d(u,v,dv)), normals[0]))
    return center, half, c, sign, best


def barycentric(points, tris):
    a, b, c = tris.transpose(1, 0, 2)
    n = np.cross(b-a, c-a)
    denom = np.einsum('ij,ij->i', n, n)
    v = np.einsum('ij,ij->i', np.cross(points-a, c-a), n)/denom
    w = np.einsum('ij,ij->i', np.cross(b-a, points-a), n)/denom
    return np.column_stack((1-v-w, v, w))


class FittedFaces:
    """Use mesh proximity to choose candidate CAD faces, then query the fitted geometry."""

    def __init__(self, guide_path, fits_path, *, compiled=True):
        with np.load(guide_path) as guide:
            self.xyz = guide['vertices']
            self.tri = guide['triangles'].astype(np.int64)
            self.owner, self.uv = guide['owners'], guide['uv']
        mesh = trimesh.Trimesh(self.xyz, self.tri, process=False)
        self.incident = mesh.vertex_faces
        self.tree = igl.AABB()
        self.tree.init(self.xyz, self.tri)
        adjacency = mesh.face_adjacency
        normals = mesh.face_normals
        crease = ((self.owner[adjacency[:,0]] != self.owner[adjacency[:,1]])
                  & (np.einsum('ij,ij->i', normals[adjacency[:,0]], normals[adjacency[:,1]]) < .999))
        self.edge_pairs = adjacency[crease]
        self.edges = mesh.face_adjacency_edges[crease].copy()
        self.edge_tree = igl.AABB()
        if len(self.edges):
            self.edge_tree.init(self.xyz, self.edges)
        inside = mesh.triangles_center[self.edge_pairs[:,1]] - self.xyz[self.edges[:,0]]
        self.convex_sign = np.where(np.einsum('ij,ij->i', inside, normals[self.edge_pairs[:,0]]) < 0, 1., -1.)
        with np.load(fits_path) as fits:
            self.faces = {}
            for idx in np.unique(self.owner):
                coef = fits[f'coef_{idx}']
                self.faces[idx] = (fits[f'center_{idx}'], fits[f'half_{idx}'], coef,
                                   cheb.chebder(coef, axis=0), cheb.chebder(coef, axis=1), fits[f'sign_{idx}'])
        self.refine = None
        if compiled:
            try:
                from fitted_projection import pack_faces, refine
            except ModuleNotFoundError as exc:
                if exc.name != 'numba':
                    raise
            else:
                self.refine = refine
                a,b,c = self.xyz[self.tri].transpose(1,0,2)
                ab, ac = b-a, c-a
                normal = np.cross(ab,ac)
                denom = np.einsum('ij,ij->i',normal,normal)
                self.bary_grad = np.stack((np.cross(ac,normal),np.cross(normal,ab)),axis=1)/denom[:,None,None]
                self.packed = pack_faces(self.faces)

    def support(self, points):
        _, nearest, seeds = self.tree.squared_distance(self.xyz, self.tri, points)
        weights = barycentric(seeds, self.xyz[self.tri[nearest]])
        support, queries = [], []
        for j, (face, bary) in enumerate(zip(nearest, weights)):
            positive = self.tri[face, bary > 1e-9]
            common = np.array([face])
            if len(positive) in (1, 2):
                common = self.incident[positive[0]]
                if len(positive) == 2:
                    common = np.intersect1d(common, self.incident[positive[1]])
                common = common[common >= 0]
            _, unique = np.unique(self.owner[common], return_index=True)
            support.extend(common[unique])
            queries.extend([j]*len(unique))
        return seeds, np.asarray(queries), np.asarray(support)

    def project(self, points, seeds, query, tri):
        current = seeds.copy()
        seed_uv = np.einsum('ij,ijk->ik', barycentric(seeds[query], self.xyz[self.tri[tri]]), self.uv[tri])
        chart = np.empty_like(seed_uv)
        groups = {}
        for idx in np.unique(self.owner[tri]):
            ids = np.flatnonzero(self.owner[tri] == idx)
            center, half, *_ = self.faces[idx]
            chart[ids] = (seed_uv[ids]-center)/half
            groups[idx] = ids
        gradients = np.empty((len(query), 3))
        surfaces = np.empty_like(gradients)
        for _ in range(4):
            for idx, ids in groups.items():
                _, _, coef, d_u, d_v, sign = self.faces[idx]
                target = current[query[ids]]
                for _ in range(2):
                    u, v = chart[ids].T
                    surface = cheb.chebval2d(u,v,coef).T
                    jac = np.stack((cheb.chebval2d(u,v,d_u).T, cheb.chebval2d(u,v,d_v).T), axis=2)
                    change = np.einsum('ijk,ik->ij', np.linalg.pinv(jac), target-surface)
                    chart[ids] += change
                u, v = chart[ids].T
                surfaces[ids] = cheb.chebval2d(u,v,coef).T
                normal = sign*np.cross(cheb.chebval2d(u,v,d_u).T, cheb.chebval2d(u,v,d_v).T)
                gradients[ids] = normal/np.linalg.norm(normal, axis=1)[:, None]
            # Locally project onto one face, or their intersection at a mesh edge/vertex.
            r = np.einsum('ij,ij->i', gradients, points[query]-surfaces)
            gram = np.zeros((len(points),3,3))
            rhs = np.zeros_like(points)
            np.add.at(gram, query, gradients[:,:,None]*gradients[:,None,:])
            np.add.at(rhs, query, gradients*r[:,None])
            current = points-np.einsum('ijk,ik->ij', np.linalg.pinv(gram, rcond=1e-4), rhs)
        pseudo = np.zeros_like(points)
        np.add.at(pseudo, query, gradients)
        return current, pseudo

    def predict(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Expected a finite N by 3 array in millimetres')
        if len(points) == 0:
            return np.empty(0)
        if self.refine is not None:
            _, nearest, seeds = self.tree.squared_distance(self.xyz,self.tri,points)
            if len(self.edges):
                sq, edge, edge_seeds = self.edge_tree.squared_distance(self.xyz,self.edges,points)
            else:
                sq, edge, edge_seeds = np.empty(0), np.empty(0,np.int64), np.empty((0,3))
            return self.refine(points,nearest,seeds,sq,edge,edge_seeds,self.xyz,self.tri,self.owner,self.uv,
                               self.incident,self.edge_pairs,self.convex_sign,self.bary_grad,*self.packed)
        seeds, query, tri = self.support(points)
        foot, normal = self.project(points, seeds, query, tri)
        delta = points-foot
        result = np.copysign(np.linalg.norm(delta, axis=1), np.einsum('ij,ij->i', delta, normal))
        if not len(self.edges):
            return result
        sq, edge, seed_on_edge = self.edge_tree.squared_distance(self.xyz, self.edges, points)
        # Expand the search near creases where tessellation error can change the nearest feature.
        near = (np.sqrt(sq) <= 2*np.linalg.norm(points-seeds, axis=1)+.005) & (np.bincount(query) <= 2)
        if not np.any(near):
            return result
        x, start, sides = points[near], seed_on_edge[near], self.edge_pairs[edge[near]]
        ids = np.arange(len(x))
        orientation = self.convex_sign[edge[near]]
        edge_foot, _ = self.project(x, start, np.repeat(ids, 2), sides.ravel())
        distance = np.linalg.norm(x-edge_foot, axis=1)
        local = [self.project(x, start, ids, side) for side in sides.T]
        signed = []
        for i, (p, n) in enumerate(local):
            other_p, other_n = local[1-i]
            allowed = orientation*np.einsum('ij,ij->i', p-other_p, other_n) <= 0
            gap = np.linalg.norm(x-p, axis=1)
            distance = np.where(allowed, np.minimum(distance, gap), distance)
            signed.append(orientation*np.einsum('ij,ij->i', x-p, n))
        result[near] = orientation*np.copysign(distance, np.maximum(*signed))
        return result


def score(values, truth, groups):
    if not np.isfinite(values).all():
        raise ValueError('Non-finite signed distance')
    error_um = 1000*abs(values-truth)
    out = {}
    for name in np.unique(groups):
        select = groups == name
        e = error_um[select]
        sign = select & (abs(truth) > 1e-6)
        out[str(name)] = {'n':len(e), 'p95_um':float(np.percentile(e,95)), 'max_um':float(e.max()),
                          'within_0p125_um':int(sum(e <= .125)),
                          'wrong_sign':int(sum(np.sign(values[sign])!=np.sign(truth[sign]))),'sign_checks':int(sign.sum())}
    return out


def run(trial, output):
    samples = np.genfromtxt(trial/'spur-cad-samples.csv', delimiter=',', skip_header=1)
    if samples.shape[1] != 7 or not np.all(samples[:,0] == 1):
        raise ValueError('Expected accepted CAD training samples')
    uv_file = trial/'spur-cad-uv.txt'
    uv = np.loadtxt(uv_file if uv_file.exists() else trial/'spur-face-uv.txt')
    rng = np.random.default_rng(104)
    arrays, validation = {}, {}
    for idx in np.unique(uv[:,0]).astype(int):
        sel = uv[:,0] == idx
        center, half, coef, sign, residual = fit_face(uv[sel,1:], samples[sel,1:4], samples[sel,4:], rng)
        for key, value in [('center',center),('half',half),('coef',coef),('sign',sign)]:
            arrays[f'{key}_{idx}'] = value
        validation[str(idx)] = residual*1000
    np.savez_compressed(output/'faces.npz', **arrays)
    fitted = FittedFaces(output/'guide.npz', output/'faces.npz')
    aux = np.genfromtxt(output/'truth-801.csv', delimiter=',', skip_header=1)
    valid = np.isfinite(aux).all(axis=1)
    train_x = np.vstack((samples[:,1:4],np.loadtxt(output/'probe-801.txt')[valid]))
    train_d = np.r_[np.zeros(len(samples)),aux[valid,0]]
    train_n = np.vstack((samples[:,4:],aux[valid,1:]))
    tangent = TangentField(train_x, train_d, train_n).eval()
    archive = np.load(trial/'spur-reference.npz')
    withheld = np.genfromtxt(output/'truth-903.csv', delimiter=',', skip_header=1)[:,0]
    if len(withheld)!=1536 or not np.isfinite(withheld).all():
        raise ValueError('Incomplete independent CAD reference')
    fresh_cad = np.genfromtxt(output/'truth-1407.csv', delimiter=',', skip_header=1)
    fresh = fresh_cad[:,0]
    if len(fresh)!=1536 or not np.isfinite(fresh).all():
        raise ValueError('Incomplete final CAD reference')
    pos = np.vstack((archive['points_mm'], np.loadtxt(output/'probe-903.txt'), np.loadtxt(output/'probe-1407.txt')))
    truth = np.r_[archive['distance_mm'],withheld,fresh]
    group = np.concatenate((archive['group'],np.repeat(['diagnostic_face','diagnostic_edge','fresh_face','fresh_edge'],768)))
    with torch.no_grad():
        global_d = np.concatenate([tangent(torch.from_numpy(p)).numpy() for p in np.array_split(pos,32)])
    report = {'surface_training_count':len(samples),'off_surface_tangent_training_count':int(valid.sum()),
              'fitted_faces':len(validation),'max_face_fit_selection_error_um':max(validation.values()),
              'diagnostic_seed':903,'final_test_seed':1407,'new_test_offset_um':50, 'models':{}}
    mesh_distance, *_ = igl.signed_distance(pos, fitted.xyz, fitted.tri, igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
    started = perf_counter()
    refined = fitted.predict(pos)
    report['fitted_batch'] = {'points':len(pos), 'seconds_excluding_setup':perf_counter()-started}
    for name, val in [('global_tangent',global_d),('triangle_mesh',mesh_distance),
                      ('fitted_faces',refined),('cad_reference',truth)]:
        scores = score(val, truth, group)
        loads = {}
        pair = archive['contact_pair']
        for key in sorted(set(pair)-{''}):
            m = np.flatnonzero(pair == key)
            loads[str(key)] = float(np.dot(1e6*archive['weights_m'][m], np.maximum(.055-val[m],0)))
        report['models'][name] = {'distances':scores,'line_load_N':loads}
    # Check actual derivatives of the saved distance function without using CAD normals to select features.
    ids = np.arange(0,len(fresh),12)
    ids = ids[np.isfinite(fresh_cad[ids,1:]).all(axis=1)]
    h = 1e-5
    x = pos[-len(fresh):][ids]
    shifts = np.concatenate((np.eye(3),-np.eye(3))) * h
    shifted = fitted.predict((x[None,:,:]+shifts[:,None,:]).reshape(-1,3)).reshape(6,len(x))
    gradient = ((shifted[:3]-shifted[3:])/(2*h)).T
    norm = np.linalg.norm(gradient,axis=1)
    cosine = np.einsum('ij,ij->i',gradient,fresh_cad[ids,1:])/np.maximum(norm,1e-15)
    angle = np.rad2deg(np.arccos(np.clip(cosine,-1,1)))
    report['fresh_normals'] = {'checks':len(ids),'fd_step_mm':h,'p95_deg':float(np.percentile(angle,95)),
                               'max_deg':float(angle.max())}
    print(json.dumps(report, indent=2))
    (output/'field-report.json').write_text(json.dumps(report, indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--step',type=Path,default=Path('results/gears/models/spur-1a.step'))
    parser.add_argument('--mesh-trial',type=Path,default=Path('build/aes-nhrep'))
    parser.add_argument('--surface-trial',type=Path,default=Path('build/aes-nhrep-parametric'))
    parser.add_argument('--prepare',type=Path,metavar='NC_CONTACT_EXE')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        if args.prepare:
            prepare(args.step,args.mesh_trial,args.prepare,args.output)
        run(args.surface_trial,args.output)
