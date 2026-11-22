"""Fit coordinate-only distance fields for the 50 mm STEP cube and audit held-out points.

Run from the repository root. Pass the NH-Rep checkout, nc_contact executable
(with its OCCT runtime on PATH) and an output directory, in that order.
Both saved TorchScript models accept batches of XYZ coordinates in mm and return mm.
"""

import argparse
from itertools import combinations, product
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from scipy.stats import qmc
from torch import nn


HALF_MM = 25.
BAND_MM = .055
GROUPS = ('face', 'edge', 'corner', 'volume')


def reference(points):
    delta = np.abs(points) - HALF_MM
    outside = np.maximum(delta, 0)
    length = np.linalg.norm(outside, axis=1)
    distance = length + np.minimum(delta.max(axis=1), 0)
    direction = np.eye(3)[delta.argmax(axis=1)]
    np.divide(outside, length[:, None], out=direction, where=length[:, None] > 0)
    return distance, direction * np.sign(points)


def dataset(seed, powers):
    rng = np.random.default_rng(seed)
    blocks, groups = [], []
    for k, power in enumerate(powers, 1):
        for axes in combinations(range(3), k):
            for signs in product((-1, 1), repeat=k):
                z = qmc.Sobol(6, scramble=True, seed=int(rng.integers(2**31))).random_base2(power)
                points = (2*z[:, :3]-1)*HALF_MM
                if k == 1:
                    directions = np.ones((len(z), 1))
                else:
                    angle = z[:, 3]*np.pi/2
                    directions = np.column_stack((np.cos(angle), np.sin(angle)))
                    if k == 3:
                        height = z[:, 4]
                        directions = np.column_stack((directions*np.sqrt(1-height[:, None]**2), height))
                radius = (2*z[:, 5]-1)*BAND_MM
                points[:, axes] = signs*(HALF_MM + radius[:, None]*directions)
                blocks.append(points)
                groups.extend([GROUPS[k-1]]*len(points))
    volume = rng.uniform(-1.5*HALF_MM, 1.5*HALF_MM, (4096, 3))
    points = np.concatenate([*blocks, volume])
    return points, np.array([*groups, *(['volume']*len(volume))])


class TangentField(nn.Module):
    def __init__(self, points, distance, gradient):
        super().__init__()
        # First-order interpolation gives each affine neuron its slope and intercept.
        _, keep = np.unique(np.round(gradient, 12), axis=0, return_index=True)
        slopes = gradient[keep]
        intercepts = distance[keep] - np.einsum('ij,ij->i', slopes, points[keep])
        self.register_buffer('slopes', torch.from_numpy(slopes.T.copy()))
        self.register_buffer('intercepts', torch.from_numpy(intercepts))

    def forward(self, xyz):
        return (xyz.double() @ self.slopes + self.intercepts).amax(dim=1)


class MillimetreField(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, xyz):
        return self.net(xyz.float()/25.)[:, 0]*25.


def fit_nhrep(upstream, points, distance, gradient, device):
    sys.path.insert(0, str(upstream.resolve()/'code'))
    from model.network import NHRepNet

    class ConvexNHRep(NHRepNet):
        def nested_cvx_output(self, values, operation, convex=True):
            # The six leaves form a single intersection at the root.
            return values.max(dim=1, keepdim=True).values

    net = ConvexNHRep(d_in=3, dims_sdf=[256, 256, 256], csg_tree=list(range(6)), n_branch=6).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    steps = 4000
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=.000005)
    x, y, n = [torch.tensor(a, dtype=torch.float32, device=device)
               for a in (points/HALF_MM, distance/HALF_MM, gradient)]
    axis = n.abs().argmax(1)
    is_face = n.abs().amax(1) > 1-1e-7
    patch = 2*axis + (n.gather(1, axis[:, None]).flatten() > 0).long()
    for _ in range(steps):
        idx = torch.randint(len(x), (512,), device=device)
        p = x[idx].requires_grad_()
        values = net(p)
        h = values[:, 0]
        slope = torch.autograd.grad(h.sum(), p, create_graph=True)[0]
        labeled = values[:, 1:].gather(1, patch[idx, None]).flatten()
        loss = ((h-y[idx]).abs().mean() + ((labeled-y[idx]).abs()*is_face[idx]).mean()
                + .1*(slope-n[idx]).square().mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
    return MillimetreField(net.eval().cpu())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'executable', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    torch.set_num_threads(1)
    train, _ = dataset(42, (10, 9, 12))
    test, group = dataset(1701, (9, 9, 10))
    y, normal = reference(train)
    truth, test_normals = reference(test)
    # Cross-check the analytic teacher against the existing STEP solid, independently by group.
    rng = np.random.default_rng(98)
    indices = np.concatenate([rng.choice(np.flatnonzero(group == g), 64, replace=False) for g in GROUPS])
    point_file = args.output/'cad-check-mm.txt'
    np.savetxt(point_file, test[indices], fmt='%.17g')
    csv = args.output/'cad-check.csv'
    subprocess.run([args.executable.resolve(), 'cad-point-distance', 'results/contact_validation/models/cube.step',
                    point_file, csv], check=True)
    cad = np.genfromtxt(csv, delimiter=',', skip_header=1)[:, 0]
    cad_error = float(np.max(abs(cad-truth[indices]))*1000)
    if not np.isfinite(cad_error) or cad_error > .002:
        raise ValueError(f'STEP and analytic reference disagree by {cad_error} micrometres')
    metrics = {'side_mm': 2*HALF_MM, 'band_um': 1000*BAND_MM, 'train_points':len(train),
               'test_points':len(test), 'cad_checks':len(indices), 'cad_max_difference_um':cad_error,
               'train_seed':42, 'test_seed':1701, 'nhrep_steps':4000, 'nhrep_batch':512,
               'teacher':'analytic signed Euclidean distance and gradient, units mm',
               'volume_domain_mm':[-37.5,37.5], 'training_device':str(device)}
    tangent = TangentField(train, y, normal)
    metrics['affine_neurons'] = int(len(tangent.intercepts))
    for name in ('tangent', 'nhrep'):
        model = tangent if name == 'tangent' else fit_nhrep(args.upstream, train, y, normal, device)
        example = torch.as_tensor(test[:3])
        with torch.no_grad():
            model(example)
            field = torch.jit.trace(model, example)
            path = args.output/f'cube-{name}.pt'
            field.save(str(path))
            eval_device = device
            restored = torch.jit.load(str(path)).eval().to(eval_device)
            values = np.concatenate([restored(torch.from_numpy(p).to(eval_device)).cpu().numpy()
                                     for p in np.array_split(test, (len(test)+127)//128)])
            # Includes six face centres, twelve edge midpoints and eight vertices.
            boundary = np.array([p for p in product((-HALF_MM, 0., HALF_MM), repeat=3) if any(p)])
            surface_max = restored(torch.from_numpy(boundary).to(eval_device)).abs().max().item()*1000
        errors = np.abs(values-truth)*1000
        assert np.isfinite(errors).all()
        metrics[name] = {'boundary_26_max_um':surface_max}
        for g in GROUPS:
            e = errors[group==g]
            ids = np.flatnonzero(group==g)[::16]
            gradients = []
            for batch in np.array_split(ids, (len(ids)+63)//64):
                x = torch.from_numpy(test[batch]).to(eval_device).requires_grad_()
                gradients.append(torch.autograd.grad(restored(x).sum(), x)[0].detach().cpu().numpy())
            gradients = np.concatenate(gradients)
            cosine = np.sum(gradients*test_normals[ids], axis=1)/np.maximum(np.linalg.norm(gradients, axis=1), 1e-12)
            angle = np.rad2deg(np.arccos(np.clip(cosine,-1,1)))
            metrics[name][g] = {'n':len(e), 'p95_um':float(np.percentile(e,95)), 'max_um':float(e.max()),
                                'within_0p125_um':int(np.sum(e<=.125)), 'normal_checks':len(ids),
                                'p95_normal_deg':float(np.percentile(angle,95))}
        print(name, metrics[name], flush=True)
    (args.output/'cube-field.json').write_text(json.dumps(metrics, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
