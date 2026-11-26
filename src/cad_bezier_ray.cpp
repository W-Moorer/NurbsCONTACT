#include "cad_bezier_ray.hpp"
#include "cad_span_index.hpp"
#include <BRepAdaptor_Surface.hxx>
#include <GeomConvert_BSplineSurfaceToBezierSurface.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom_BezierSurface.hxx>
#include <TopExp.hxx>
#include <TopoDS.hxx>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace nc::cad {
namespace {
using Four = std::array<double, 4>;
Four homogeneous(const gp_XYZ &xyz, double w) { return {xyz.X(), xyz.Y(), xyz.Z(), w}; }
gp_XYZ xyz(const Four &p) { return {p[0], p[1], p[2]}; }
gp_XYZ absolute(const gp_XYZ &v) { return {std::abs(v.X()), std::abs(v.Y()), std::abs(v.Z())}; }
struct Range {
    Four mid{}, radius{};
    void set(const Four &low, const Four &high) {
        for (int i = 0; i < 4; ++i) {
            mid[i] = (high[i] + low[i]) * .5;
            radius[i] = (high[i] - low[i]) * .5;
        }
    }
};
} // namespace

struct BezierRayAtlas::Span {
    std::size_t face;
    double u0, u1, v0, v1;
    gp_XYZ anchor;
    Four poles[4][4], center{};
    Range slopes[2];
    gp_XYZ lo, hi, axis;
    double low, high, min_weight = 1, max_weight = 1;

    Span() = default;
    Span(const Handle(Geom_BezierSurface) & patch, std::size_t id, double u0, double u1, double v0, double v1)
        : face(id), u0(u0), u1(u1), v0(v0), v1(v1), anchor(patch->Pole(1, 1).XYZ()), lo(anchor), hi(anchor) {
        patch->Increase(3, 3);
        gp_Pnt p;
        gp_Vec du, dv;
        patch->D1(.5, .5, p, du, dv);
        axis = du.XYZ().Crossed(dv.XYZ());
        if (axis.Modulus() > 0)
            axis.Normalize();
        low = high = 0;
        const double scale = patch->Weight(1, 1);
        constexpr double basis[]{.125, .375, .375, .125};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                const gp_XYZ pos = patch->Pole(i + 1, j + 1).XYZ(), rel = pos - anchor;
                const double w = patch->Weight(i + 1, j + 1) / scale;
                if (!(std::isfinite(w) && w > 0))
                    throw Standard_Failure("invalid Bezier weight");
                min_weight = std::min(min_weight, w);
                max_weight = std::max(max_weight, w);
                poles[i][j] = homogeneous(rel * w, w);
                for (int c = 0; c < 4; ++c)
                    center[c] += basis[i] * basis[j] * poles[i][j][c];
                for (int k = 1; k <= 3; ++k) {
                    lo.SetCoord(k, std::min(lo.Coord(k), pos.Coord(k)));
                    hi.SetCoord(k, std::max(hi.Coord(k), pos.Coord(k)));
                }
                const double h = axis.Dot(rel);
                low = std::min(low, h);
                high = std::max(high, h);
            }
        for (int dir = 0; dir < 2; ++dir) {
            Four lower, upper;
            lower.fill(std::numeric_limits<double>::infinity());
            upper.fill(-std::numeric_limits<double>::infinity());
            for (int i = 0; i < (dir ? 4 : 3); ++i)
                for (int j = 0; j < (dir ? 3 : 4); ++j)
                    for (int k = 0; k < 4; ++k) {
                        const double h = 3 * (poles[i + (dir == 0)][j + (dir == 1)][k] - poles[i][j][k]);
                        lower[k] = std::min(lower[k], h);
                        upper[k] = std::max(upper[k], h);
                    }
            slopes[dir].set(lower, upper);
        }
    }
    // Numerator of the rational surface relative to the ray origin, and its
    // two derivatives. Projection perpendicular to the ray removes its length.
    void jet(double u, double v, const gp_XYZ &origin, gp_XYZ &value, gp_XYZ &du, gp_XYZ &dv,
             double &w) const {
        const auto basis = [](double x, double *b, double *d) {
            const double y = 1 - x, a = y * y, c = x * x, xy = x * y;
            b[0] = a * y;
            b[1] = 3 * a * x;
            b[2] = 3 * y * c;
            b[3] = c * x;
            d[0] = -3 * a;
            d[1] = 3 * a - 6 * xy;
            d[2] = 6 * xy - 3 * c;
            d[3] = 3 * c;
        };
        double bu[4], bv[4], duv[4], dvv[4];
        basis(u, bu, duv);
        basis(v, bv, dvv);
        Four q{}, a{}, b{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                const auto &h = poles[i][j];
                const double f = bu[i] * bv[j], s = duv[i] * bv[j], t = bu[i] * dvv[j];
                for (int k = 0; k < 4; ++k) {
                    q[k] += f * h[k];
                    a[k] += s * h[k];
                    b[k] += t * h[k];
                }
            }
        value = xyz(q) - origin * q[3];
        du = xyz(a) - origin * a[3];
        dv = xyz(b) - origin * b[3];
        w = q[3];
    }
    // With positive weights the signed distance numerator lies in the
    // convex hull of these 16 Bernstein coefficients. One strict sign
    // excludes a crossing of the infinite line anywhere in this span.
    bool excludes_line(const gp_XYZ& row,const gp_XYZ& origin,double tol) const {
        const double projection=row.Dot(origin);
        const double length=row.Modulus();
        bool positive=true,negative=true;
        for (const auto& strip:poles) for (const auto& h:strip) {
            const double value=row.Dot(xyz(h))-h[3]*projection;
            const double pad=length*h[3]*tol+64*std::numeric_limits<double>::epsilon()*
                (absolute(row).Dot(absolute(xyz(h))+absolute(origin)*h[3])+1.);
            positive &= value>pad;
            negative &= value< -pad;
            if (!positive && !negative) return false;
        }
        return true;
    }
    // Every projected root z satisfies z = c-R*f(c)+(I-R*J)*(z-c).
    // Intersect its enclosure with the previous box, also allowing the face
    // tolerance. An empty box is the only negative answer issued here.
    bool root_box(const gp_XYZ& origin,const gp_XYZ (&rows)[2],const double (&errors)[2][2],
                  double tol,std::array<double,2>& lower,std::array<double,2>& upper) const {
        for (int pass=0;pass<4;++pass) {
            const double x=(lower[0]+upper[0])*.5,y=(lower[1]+upper[1])*.5;
            const double hx=(upper[0]-lower[0])*.5,hy=(upper[1]-lower[1])*.5;
            gp_XYZ value,du,dv;
            double w;
            jet(x,y,origin,value,du,dv,w);
            for (int k=0;k<2;++k) {
                const double center=(k==0 ? x : y)-rows[k].Dot(value);
                const double radius=errors[k][0]*hx+errors[k][1]*hy+
                    rows[k].Modulus()*max_weight*tol+1e-11*(1+std::abs(center));
                lower[k]=std::max(lower[k],center-radius);
                upper[k]=std::min(upper[k],center+radius);
                if (lower[k]>upper[k]) return false;
            }
        }
        return true;
    }
    bool intersect(const gp_XYZ &point, const gp_XYZ &d, double end, double tol,
                   std::optional<BezierRayHit> &hit,bool relocate,const std::array<double,2>* previous_uv) const {
        const auto origin = point - anchor;
        const double p = axis.Dot(origin), q = p + axis.Dot(d) * end;
        if (std::min(p, q) > high + tol || std::max(p, q) < low - tol)
            return true;
        const auto derivative = [&](const Four &h) { return xyz(h) - origin * h[3]; };
        const auto a = derivative(slopes[0].mid), b = derivative(slopes[1].mid);
        const auto row_u = b.Crossed(d), row_v = d.Crossed(a);
        const double det = a.Dot(row_u);
        if (std::abs(det) <= 1e-10 * a.Modulus() * b.Modulus() || !std::isfinite(1. / det))
            return false;
        const gp_XYZ rows[]{row_u / det, row_v / det};
        double contraction = 0;
        const auto f = derivative(center);
        double xy[]{.5, .5};
        double errors[2][2]{};
        for (int k = 0; k < 2; ++k) {
            double r = 0;
            for (int j=0;j<2;++j) {
                const auto& slope=slopes[j];
                const double variation=absolute(rows[k]).Dot(xyz(slope.radius))+
                                       std::abs(rows[k].Dot(origin))*slope.radius[3];
                const double midpoint_error=std::abs((k==j ? 1. : 0.)-rows[k].Dot(derivative(slope.mid)));
                errors[k][j]=variation+midpoint_error+1e-10*(1+variation);
                r+=variation;
            }
            r += 1e-10 * (1 + r);
            xy[k] -= rows[k].Dot(f);
            const double pad = tol * rows[k].Modulus() * max_weight;
            if (xy[k] + r * .5 < -pad || xy[k] - r * .5 > 1 + pad)
                return true;
            contraction = std::max(contraction, r);
        }
        if (!(contraction < .9))
            return false;
        if (xy[0]<=0 || xy[0]>=1 || xy[1]<=0 || xy[1]>=1) {
            if (!relocate) return false;
            for (int k=0;k<2;++k)
                if ((xy[k]<=0 || xy[k]>=1) && excludes_line(rows[k],origin,tol)) return true;
            std::array<double,2> lower{0.,0.},upper{1.,1.};
            if (!root_box(origin,rows,errors,tol,lower,upper)) return true;
            // The spatial tree also visits adjacent spans on the same face.
            // Express the last face UV in this span, including an adjacent
            // seed near its edge. Every root still needs the original proof.
            if (previous_uv) {
                const double u=((*previous_uv)[0]-u0)/(u1-u0),v=((*previous_uv)[1]-v0)/(v1-v0);
                if (u>=-1 && u<=2 && v>=-1 && v<=2) {xy[0]=u;xy[1]=v;}
            }
            for (int k=0;k<2;++k) xy[k]=std::clamp(xy[k],lower[k],upper[k]);
        }
        for (int iter = 0; iter < 8; ++iter) {
            gp_XYZ value, du, dv;
            double w;
            jet(xy[0], xy[1], origin, value, du, dv, w);
            if (!(w > 0))
                return false;
            const double t = d.Dot(value) / w;
            const auto residual = value - d * (w * t);
            if (residual.Modulus() < 1e-9 * w) {
                // The fixed-point map has at most one root on this span.
                // Its a posteriori radius must fit strictly inside the span.
                const double radius =
                    (std::max(std::abs(rows[0].Dot(residual)), std::abs(rows[1].Dot(residual))) + 1e-12) /
                    (1 - contraction);
                if (radius >= std::min({xy[0], 1 - xy[0], xy[1], 1 - xy[1]}))
                    return false;
                const auto recovered = origin + d * t;
                double error = residual.Modulus();
                for (const auto &slope : slopes) {
                    const auto mid = xyz(slope.mid) - recovered * slope.mid[3];
                    error += radius * (mid.Modulus() + xyz(slope.radius).Modulus() +
                                       recovered.Modulus() * slope.radius[3]);
                }
                error = error / min_weight + tol;
                if (t < -tol - error || t > end + error)
                    return true;
                if (t <= error || t >= end - error || hit)
                    return false;
                // Rational tangents require subtracting the weight derivative
                // along the ray. Evaluate at the recovered point as origin.
                jet(xy[0], xy[1], recovered, value, du, dv, w);
                auto n = du.Crossed(dv);
                if (!(n.Modulus() > 0))
                    return false;
                n.Normalize();
                hit = BezierRayHit{t, u0 + xy[0] * (u1 - u0), v0 + xy[1] * (v1 - v0), n};
                return true;
            }
            const auto c = dv.Crossed(d), e = d.Crossed(du);
            const double delta = du.Dot(c);
            if (delta == 0)
                return false;
            xy[0] -= residual.Dot(c) / delta;
            xy[1] -= residual.Dot(e) / delta;
            if (!(xy[0] > 0 && xy[0] < 1 && xy[1] > 0 && xy[1] < 1)) {
                std::array<double,2> lower{0.,0.},upper{1.,1.};
                if (relocate && !root_box(origin,rows,errors,tol,lower,upper)) return true;
                return false;
            }
        }
        return false;
    }
};

struct BezierRayAtlas::Part {
    struct Box {
        gp_XYZ lo, hi;
    };
    std::vector<Span> spans;
    std::vector<Box> boxes;
    std::vector<std::size_t> offsets;
    std::vector<SpanIndex> indices;
};
BezierRayAtlas::BezierRayAtlas(const TopoDS_Shape &shape, PatchAssets &assets,bool relocate):relocate_(relocate) {
    TopTools_IndexedMapOfShape faces;
    TopExp::MapShapes(shape, TopAbs_FACE, faces);
    auto part = std::make_shared<Part>();
    part->spans =
        assets.table<Span>("bezier-rays-v2", std::numeric_limits<std::size_t>::max() / sizeof(Span), [&] {
            std::vector<Span> rows;
            for (int f = 1; f <= faces.Extent(); ++f) {
                const auto start = rows.size();
                try {
                    const BRepAdaptor_Surface s(TopoDS::Face(faces(f)));
                    if (s.GetType() == GeomAbs_BezierSurface) {
                        const auto patch = Handle(Geom_BezierSurface)::DownCast(s.Bezier()->Copy());
                        if (patch->UDegree() <= 3 && patch->VDegree() <= 3)
                            rows.emplace_back(patch, f, 0., 1., 0., 1.);
                    } else if (s.GetType() == GeomAbs_BSplineSurface) {
                        const auto bs = s.BSpline();
                        if (bs->UDegree() > 3 || bs->VDegree() > 3 || bs->IsUPeriodic() || bs->IsVPeriodic())
                            continue;
                        GeomConvert_BSplineSurfaceToBezierSurface net(bs);
                        TColStd_Array1OfReal u(1, net.NbUPatches() + 1), v(1, net.NbVPatches() + 1);
                        net.UKnots(u);
                        net.VKnots(v);
                        for (int i = 1; i < u.Upper(); ++i)
                            for (int j = 1; j < v.Upper(); ++j)
                                rows.emplace_back(net.Patch(i, j), f, u(i), u(i + 1), v(j), v(j + 1));
                    }
                } catch (const Standard_Failure &) {
                    rows.erase(rows.begin() + start, rows.end());
                }
            }
            return rows;
        });
    part->offsets.resize(faces.Extent() + 1);
    for (const auto &span : part->spans) {
        ++part->offsets.at(span.face);
        part->boxes.push_back({span.lo, span.hi});
    }
    std::partial_sum(part->offsets.begin(), part->offsets.end(), part->offsets.begin());
    part->indices.resize(faces.Extent());
    for (std::size_t i=0;i<part->indices.size();++i) {
        const auto first=part->offsets[i];
        part->indices[i].build(part->offsets[i+1]-first,[&](auto j)->const Part::Box& {return part->boxes[first+j];});
    }
    part_ = std::move(part);
}
bool BezierRayAtlas::query(std::size_t face, const gp_XYZ &point, const gp_XYZ &d, double end, double tol,
                           std::optional<BezierRayHit> &hit, std::size_t *candidate_spans,
                           const std::array<double,2>* previous_uv) const {
    hit.reset();
    const auto first = part_->offsets.at(face - 1), last = part_->offsets.at(face);
    if (first == last)
        return false;
    // Extend both ends for the native intersector's near-boundary decisions.
    const auto p = point - d * tol, q = point + d * (end + tol);
    const double margin = tol + 64 * std::numeric_limits<double>::epsilon() * (point.Modulus() + end + 1);
    double lo[3], hi[3];
    for (int k = 0; k < 3; ++k) {
        lo[k] = std::min(p.Coord(k + 1), q.Coord(k + 1)) - margin;
        hi[k] = std::max(p.Coord(k + 1), q.Coord(k + 1)) + margin;
    }
    return part_->indices[face-1].query({lo[0],lo[1],lo[2]}, {hi[0],hi[1],hi[2]}, [&](auto local) {
        const auto i=first+local;
        const auto &box = part_->boxes[i];
        if (lo[0] > box.hi.X() || hi[0] < box.lo.X() || lo[1] > box.hi.Y() || hi[1] < box.lo.Y() ||
            lo[2] > box.hi.Z() || hi[2] < box.lo.Z())
            return true;
        if (candidate_spans)
            ++*candidate_spans;
        return part_->spans[i].intersect(point, d, end, margin, hit,relocate_,previous_uv);
    });
}
} // namespace nc::cad
