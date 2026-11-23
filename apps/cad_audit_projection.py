"""Batched closest points on regular CAD faces used by the Gmsh mesh audit.

Only polynomial B-splines with a strictly convex squared-distance objective
on the whole parameter rectangle use the local iteration. Other faces and
unresolved samples retain the ordinary Gmsh projection.
"""

import gmsh
import numpy as np


def derivative(poles, knots, degree, axis):
    """Control net of a first derivative with respect to its knot parameter."""
    net = np.moveaxis(poles, axis, 0)
    widths = knots[degree + 1 : degree + len(net)] - knots[1 : len(net)]
    if degree < 1 or np.any(widths <= 0):
        raise ValueError("singular spline derivative")
    return np.moveaxis(np.diff(net, axis=0) * (degree / widths)[:, None, None], 0, axis)


def bounds(net):
    flat = net.reshape(-1, 3)
    return flat.min(axis=0), flat.max(axis=0)


def dot_range(a, b):
    terms = np.stack([x * y for x in a for y in b])
    return terms.min(axis=0).sum(), terms.max(axis=0).sum()


def squared_lower(box):
    low, high = box
    return np.square(np.maximum(np.maximum(low, -high), 0.0)).sum()


class ConvexProjection:
    def __init__(self, tag, spline, subdomain=False):
        pu, pv = spline.UDegree(), spline.VDegree()
        if (
            min(pu, pv) < 2
            or spline.IsURational()
            or spline.IsVRational()
            or spline.IsUPeriodic()
            or spline.IsVPeriodic()
        ):
            raise ValueError("requires a nonperiodic polynomial patch")
        ku, kv = np.array(list(spline.UKnotSequence())), np.array(list(spline.VKnotSequence()))
        for knots, degree in ((ku, pu), (kv, pv)):
            _, mult = np.unique(knots, return_counts=True)
            if mult[0] != degree + 1 or mult[-1] != degree + 1 or np.any(mult[1:-1] >= degree):
                raise ValueError("requires a clamped continuously differentiable patch")
        poles = np.array(
            [
                [spline.Pole(i + 1, j + 1).Coord() for j in range(spline.NbVPoles())]
                for i in range(spline.NbUPoles())
            ]
        )
        lo, hi = np.array([ku[pu], kv[pv]]), np.array([ku[-1 - pu], kv[-1 - pv]])
        domain = gmsh.model.getParametrizationBounds(2, tag)
        if not subdomain and not np.allclose(domain, [lo, hi], atol=1e-12, rtol=0):
            raise ValueError("trim range differs from spline bounds")
        # Verify face mapping across the parameter domain, including placement.
        gu = [ku[i + 1 : i + pu + 1].mean() for i in range(len(poles))]
        gv = [kv[j + 1 : j + pv + 1].mean() for j in range(poles.shape[1])]
        uv = np.array([(u, v) for u in gu for v in gv])
        cad = gmsh.model.getValue(2, tag, uv.ravel()).reshape(-1, 3)
        occ = np.array([spline.Value(u, v).Coord() for u, v in uv])
        if np.max(np.abs(cad - occ)) > 1e-10:
            raise ValueError("STEP and Gmsh parameterizations do not match")
        self.origin = cad[len(cad) // 2]
        self.mid = (lo + hi) * 0.5
        self.lo, self.hi = lo, hi
        tangent = gmsh.model.getDerivative(2, tag, self.mid).reshape(2, 3)
        ex = tangent[0] / np.linalg.norm(tangent[0])
        ez = np.cross(ex, tangent[1])
        if not np.linalg.norm(ez) > 0:
            raise ValueError("singular patch")
        ez /= np.linalg.norm(ez)
        self.basis = np.stack([ex, np.cross(ez, ex), ez], axis=1)
        local = (poles - self.origin) @ self.basis
        low, high = bounds(local)
        # The checked query box includes rounding of the Gmsh vertices.
        pad = 1e-9 + 1e-12 * np.max(np.abs(poles))
        self.hull = (low - pad, high + pad)
        halo = 0.025 * np.min(np.linalg.norm(tangent, axis=1) * (hi - lo)) if subdomain else 0.0
        self.low, self.high = self.hull[0] - halo, self.hull[1] + halo
        delta = high - low + 2 * pad + halo
        difference = (-delta, delta)
        du, dv = derivative(local, ku, pu, 0), derivative(local, kv, pv, 1)
        uu = derivative(du, ku[1:-1], pu - 1, 0)
        vv = derivative(dv, kv[1:-1], pv - 1, 1)
        uv_net = derivative(du, kv, pv, 1)
        # Hessian of |S(u,v)-x|²/2 is JᵀJ + (S-x)·D²S. Every spline
        # (including each derivative spline) lies in its control hull.
        hu = squared_lower(bounds(du)) + dot_range(difference, bounds(uu))[0]
        hv = squared_lower(bounds(dv)) + dot_range(difference, bounds(vv))[0]
        mixed = np.array(dot_range(bounds(du), bounds(dv))) + dot_range(difference, bounds(uv_net))
        cross_bound = np.abs(mixed).max()
        if not (hu > 0 and hv > 0 and hu * hv > (1.0 + 1e-8) * cross_bound**2):
            raise ValueError("whole-patch distance Hessian is not bounded positive")
        self.gradient_scale = 1 / np.sqrt([hu, hv])
        eigen_lower = 1.0 - cross_bound / np.sqrt(hu * hv)

        def upper_norm(net):
            return np.linalg.norm(net, axis=2).max()

        umax, vmax = upper_norm(du), upper_norm(dv)
        tangent_cross = max(abs(x) for x in dot_range(bounds(du), bounds(dv)))
        cross2 = squared_lower(bounds(du)) * squared_lower(bounds(dv)) - tangent_cross**2
        if cross2 <= 0:
            raise ValueError("unresolved surface normal")
        normal_derivatives = np.array(
            [
                upper_norm(uu) * vmax + umax * upper_norm(uv_net),
                upper_norm(uv_net) * vmax + umax * upper_norm(vv),
            ]
        ) / np.sqrt(cross2)
        # Strong convexity bounds parameter error by gradient norm divided by
        # its eigenvalue lower bound (in the diagonal-scaled coordinates).
        # Bound both the resulting CAD position error (mm) and normal change.
        rate_p = np.linalg.norm(np.array([umax, vmax]) * self.gradient_scale) / eigen_lower
        rate_n = np.linalg.norm(normal_derivatives * self.gradient_scale) / eigen_lower
        self.gradient_limit = min(1e-8 / rate_p, 1e-8 / max(1e-30, rate_n))
        self.tag = tag
        self.guess = np.linalg.pinv(tangent)
        self.center = gmsh.model.getValue(2, tag, self.mid)

    def solve(self, points):
        local = (points - self.origin) @ self.basis
        eligible = np.all((local >= self.low) & (local <= self.high), axis=1)
        uv = np.clip(self.mid + (points - self.center) @ self.guess, self.lo, self.hi)
        pending = np.flatnonzero(eligible)
        accepted = np.zeros(len(points), dtype=bool)
        for _ in range(8):
            if not len(pending):
                break
            params = uv[pending]
            xyz = gmsh.model.getValue(2, self.tag, params.ravel()).reshape(-1, 3)
            jac = gmsh.model.getDerivative(2, self.tag, params.ravel()).reshape(-1, 2, 3)
            a, b = jac[:, 0], jac[:, 1]
            dot = lambda x, y: np.einsum("ij,ij->i", x, y)
            aa, ab, bb = dot(a, a), dot(a, b), dot(b, b)
            residual = xyz - points[pending]
            ra, rb = dot(a, residual), dot(b, residual)
            det = aa * bb - ab * ab
            change = np.column_stack([(bb * ra - ab * rb) / det, (aa * rb - ab * ra) / det])
            trial = np.clip(params - change, self.lo, self.hi)
            # Require stationarity, also on a boundary. A projected step alone
            # can stall at a corner without solving the two gradient equations.
            stationary = (
                np.hypot(ra * self.gradient_scale[0], rb * self.gradient_scale[1]) <= self.gradient_limit
            )
            accepted[pending[stationary]] = True
            pending = pending[~stationary]
            uv[pending] = trial[~stationary]
        xyz = np.empty_like(points)
        if accepted.any():
            xyz[accepted] = gmsh.model.getValue(2, self.tag, uv[accepted].ravel()).reshape(-1, 3)
        return xyz, uv, accepted


class FaceProjection:
    def __init__(self, tag, spline):
        self.tag = tag
        try:
            self.patches = [ConvexProjection(tag, spline)]
            return
        except ValueError:
            pass
        low, high = gmsh.model.getParametrizationBounds(2, tag)
        u0, u1, v0, v1 = spline.Bounds()
        if not np.allclose([low, high], [[u0, v0], [u1, v1]], atol=1e-12, rtol=0):
            raise ValueError("requires the full underlying parameter rectangle")
        if spline.IsURational() or spline.IsVRational() or spline.IsUPeriodic() or spline.IsVPeriodic():
            raise ValueError("requires a nonperiodic polynomial patch")
        # Split along the direction in which the unit tangent varies most.
        probes = np.array([low + 0.2 * (high - low), (low + high) * 0.5, high - 0.2 * (high - low)])
        tangents = gmsh.model.getDerivative(2, tag, probes.ravel()).reshape(-1, 2, 3)
        tangents /= np.linalg.norm(tangents, axis=2)[:, :, None]
        axis = int(np.argmax(np.ptp(tangents, axis=0).sum(axis=1)))
        cuts = np.linspace(low[axis], high[axis], 9)
        self.patches = []
        pending = [(left, right, 0) for left, right in zip(cuts[:-1], cuts[1:])]
        while pending:
            left, right, level = pending.pop()
            section = spline.Copy()
            limits = [u0, u1, v0, v1]
            limits[2 * axis : 2 * axis + 2] = [left, right]
            section.Segment(*limits)
            try:
                self.patches.append(ConvexProjection(tag, section, subdomain=True))
            except ValueError:
                if level == 2:
                    raise
                mid = (left + right) * 0.5
                pending.extend([(left, mid, level + 1), (mid, right, level + 1)])

    def project(self, points):
        if len(self.patches) == 1:
            xyz, uv, accepted = self.patches[0].solve(points)
        else:
            # Bound distance to every other subpatch before accepting a local
            # stationary point as the minimum over the entire face.
            distances = []
            for patch in self.patches:
                local = (points - patch.origin) @ patch.basis
                dlo, dhi = patch.hull
                outside = np.maximum(np.maximum(dlo - local, local - dhi), 0.0)
                distances.append(np.einsum("ij,ij->i", outside, outside))
            distances = np.array(distances)
            nearest = np.argmin(distances, axis=0)
            second = np.partition(distances, 1, axis=0)[1]
            xyz, uv = np.empty_like(points), np.empty((len(points), 2))
            accepted = np.zeros(len(points), dtype=bool)
            for i, patch in enumerate(self.patches):
                ids = np.flatnonzero(nearest == i)
                if not len(ids):
                    continue
                coords, params, good = patch.solve(points[ids])
                ids = ids[good]
                xyz[ids], uv[ids] = coords[good], params[good]
                distance = np.linalg.norm(xyz[ids] - points[ids], axis=1)
                accepted[ids] = (distance + 1e-9) ** 2 < second[ids]
        fallback = ~accepted
        if fallback.any():
            other, params = gmsh.model.getClosestPoint(2, self.tag, points[fallback].ravel())
            xyz[fallback], uv[fallback] = other.reshape(-1, 3), params.reshape(-1, 2)
        return xyz, uv.ravel(), int(fallback.sum())


def projectors(step):
    if not any(gmsh.model.getType(2, tag) == "BSpline surface" for _, tag in gmsh.model.getEntities(2)):
        return {}
    try:
        from OCP.BRep import BRep_Tool
        from OCP.Geom import Geom_BSplineSurface
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
    except ImportError:
        return {}
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step.resolve())) != IFSelect_RetDone or not reader.TransferRoots():
        return {}
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_FACE)
    result = {}
    for _, tag in gmsh.model.getEntities(2):
        if not explorer.More():
            return {}
        shape = TopoDS.Face_s(explorer.Current())
        explorer.Next()
        if gmsh.model.getType(2, tag) != "BSpline surface":
            continue
        spline = BRep_Tool.Surface_s(shape)
        if isinstance(spline, Geom_BSplineSurface):
            try:
                result[tag] = FaceProjection(tag, spline)
            except (ValueError, RuntimeError):
                pass
    return result
