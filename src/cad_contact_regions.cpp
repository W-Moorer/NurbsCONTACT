#include "nurbscontact/cad_contact_regions.hpp"

#include <BRepAdaptor_Surface.hxx>
#include <BRepBndLib.hxx>
#include <BRepClass_FaceClassifier.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <Extrema_FuncExtSS.hxx>
#include <Geom2dAPI_InterCurveCurve.hxx>
#include <Geom2d_Line.hxx>
#include <Geom2d_TrimmedCurve.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <NCollection_UBTreeFiller.hxx>
#include <Precision.hxx>
#include <Standard_Failure.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <gp_Dir2d.hxx>
#include <gp_Vec2d.hxx>
#include <gp_Ax1.hxx>
#include <gp_Ax3.hxx>
#include <gp_Pnt2d.hxx>
#include <math_FunctionSetRoot.hxx>

#include <algorithm>
#include <map>
#include <stdexcept>

namespace nc {
namespace {

using namespace dynamics;
using FaceTree = NCollection_UBTree<std::size_t, Bnd_Box>;
constexpr double mm = .001;

struct Face {
    TopoDS_Face shape;
    Bnd_Box bounds;
    gp_Pnt center;
    double radius_mm{};
};

using FacePair = std::pair<std::size_t, std::size_t>;

struct FaceExclusion {
    gp_Vec axis;
    gp_Vec local_axis;
    gp_Pnt moved_center;
    double gap_mm;
};

gp_Vec local_axis(const gp_Vec& axis, const gp_Trsf& inverse) {
    return axis.Transformed(inverse);
}

bool stays_separated(const FaceExclusion& record, const Face& b, const gp_Trsf& pose,
                     const gp_Trsf& inverse, double cutoff_mm) {
    const gp_Pnt center = b.center.Transformed(pose);
    const double projected_translation = gp_Vec(record.moved_center, center).Dot(record.axis);
    const double angular_loss = b.radius_mm * (local_axis(record.axis, inverse) - record.local_axis).Magnitude();
    return record.gap_mm + projected_translation - angular_loss > cutoff_mm + Precision::Confusion();
}

std::optional<FaceExclusion> projected_exclusion(const TopoDS_Face& a, const Face& local_b,
    const TopoDS_Face& moved_b, const gp_Trsf& pose, const gp_Trsf& inverse,
    const gp_Pnt& pa, const gp_Pnt& pb, double cutoff_mm) {
    const gp_Dir axis(gp_Vec{pa, pb});
    gp_Trsf alignment;
    alignment.SetTransformation(gp_Ax3(gp_Pnt{}, axis));
    const TopLoc_Location placement(alignment);
    // Bound the source geometry directly in coordinates with z along the selected axis.
    Bnd_Box ba, bb;
    BRepBndLib::AddOptimal(a.Moved(placement), ba, false, true);
    BRepBndLib::AddOptimal(moved_b.Moved(placement), bb, false, true);
    if (ba.IsVoid() || ba.IsOpen() || bb.IsVoid() || bb.IsOpen()) return std::nullopt;
    const double gap = bb.CornerMin().Z() - ba.CornerMax().Z();
    if (!(gap > cutoff_mm + Precision::Confusion())) return std::nullopt;
    const gp_Vec direction(axis);
    return FaceExclusion{direction, local_axis(direction, inverse), local_b.center.Transformed(pose), gap};
}

class NearBox final : public FaceTree::Selector {
public:
    explicit NearBox(Bnd_Box box) : box_(std::move(box)) {}

    bool Reject(const Bnd_Box& box) const override { return box_.IsOut(box); }
    bool Accept(const std::size_t& face) override {
        faces.push_back(face);
        return true;
    }

    std::vector<std::size_t> faces;

private:
    Bnd_Box box_;
};

CadFaceWitness face_witness(const BRepExtrema_DistShapeShape& query, int solution, std::size_t face, bool first) {
    CadFaceWitness witness;
    witness.face = face;
    const auto kind = first ? query.SupportTypeShape1(solution) : query.SupportTypeShape2(solution);
    if (kind == BRepExtrema_IsInFace) {
        witness.support = FeatureKind::face;
        auto& [u, v] = witness.uv.emplace();
        if (first) query.ParOnFaceS1(solution, u, v);
        else query.ParOnFaceS2(solution, u, v);
    } else {
        witness.support = kind == BRepExtrema_IsOnEdge ? FeatureKind::edge : FeatureKind::vertex;
    }
    return witness;
}

Vec point(const gp_Pnt& p) { return {p.X() * mm, p.Y() * mm, p.Z() * mm}; }

bool faces_direction(const TopoDS_Face& face, const CadFaceWitness& witness, Vec direction) {
    if (!witness.uv) return false;
    gp_Pnt origin;
    gp_Vec du, dv;
    BRepAdaptor_Surface(face).D1((*witness.uv)[0], (*witness.uv)[1], origin, du, dv);
    const gp_Vec n = du.Crossed(dv);
    const Vec normal{n.X(), n.Y(), n.Z()};
    const double norm_n = length(normal);
    if (!std::isfinite(norm_n) || norm_n == 0) return false;
    const double sign = face.Orientation() == TopAbs_REVERSED ? -1. : 1.;
    return sign * dot3(normal, direction) / norm_n >= 1. - 1e-6;
}

TopLoc_Location relative_placement(ContactPose pose, double cutoff_m) {
    const auto finite = [](Vec v) { return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); }); };
    const double axis_length = length(pose.axis);
    if (!(cutoff_m >= 0) || !std::isfinite(cutoff_m / mm) || !std::isfinite(pose.angle) ||
        !finite(pose.origin) || !finite(scale(pose.origin, 1/mm)) || !(axis_length > 0) || !std::isfinite(axis_length)) {
        throw std::invalid_argument("invalid CAD contact region query");
    }
    gp_Trsf transform;
    if (pose.angle != 0) {
        transform.SetRotation(gp_Ax1{gp_Pnt{}, gp_Dir{pose.axis[0], pose.axis[1], pose.axis[2]}}, pose.angle);
    }
    transform.SetTranslationPart(gp_Vec{pose.origin[0] / mm, pose.origin[1] / mm, pose.origin[2] / mm});
    return TopLoc_Location(transform);
}

std::optional<CadContactCandidate> track_point(const TopoDS_Face& fa, const TopoDS_Face& fb,
                                                const CadContactSeed& seed, double cutoff) {
    const BRepAdaptor_Surface a(fa), b(fb);
    const std::array lower{a.FirstUParameter(), a.FirstVParameter(), b.FirstUParameter(), b.FirstVParameter()};
    const std::array upper{a.LastUParameter(), a.LastVParameter(), b.LastUParameter(), b.LastVParameter()};
    const std::array parameters{seed.uv_a[0], seed.uv_a[1], seed.uv_b[0], seed.uv_b[1]};
    math_Vector initial(1, 4), lo(1, 4), hi(1, 4), tolerance(1, 4);
    for (int i = 1; i <= 4; ++i) {
        const auto k = static_cast<std::size_t>(i - 1);
        if (!std::isfinite(parameters[k]) || !Interval{lower[k], upper[k]}.contains(parameters[k])) return std::nullopt;
        initial(i) = parameters[k];
        lo(i) = lower[k];
        hi(i) = upper[k];
        tolerance(i) = Precision::PConfusion();
    }
    Extrema_FuncExtSS function(a, b);
    math_FunctionSetRoot root(function, tolerance, 32);
    root.Perform(function, initial, lo, hi);
    if (!root.IsDone()) return std::nullopt;
    const auto& uv = root.Root();
    if (BRepClass_FaceClassifier(fa, gp_Pnt2d{uv(1), uv(2)}, Precision::PConfusion()).State() != TopAbs_IN ||
        BRepClass_FaceClassifier(fb, gp_Pnt2d{uv(3), uv(4)}, Precision::PConfusion()).State() != TopAbs_IN) {
        return std::nullopt;
    }
    const Vec p = point(a.Value(uv(1), uv(2))), q = point(b.Value(uv(3), uv(4)));
    const Vec delta = sub(q, p);
    const double distance = length(delta);
    if (!(distance > 1e-12 && distance <= cutoff)) return std::nullopt;
    const Vec direction = scale(delta, 1 / distance);
    const CadFaceWitness wa{seed.face_a, FeatureKind::face, std::array{uv(1), uv(2)}};
    const CadFaceWitness wb{seed.face_b, FeatureKind::face, std::array{uv(3), uv(4)}};
    if (!faces_direction(fa, wa, direction) || !faces_direction(fb, wb, scale(direction, -1))) return std::nullopt;
    return CadContactCandidate{wa, wb, {p, q, distance}, true};
}

struct FaceSegment {
    gp_Pnt2d start;
    gp_Vec2d vector;
    gp_Pnt2d at(double t) const { return start.Translated(vector.Multiplied(t)); }
};

std::optional<FaceSegment> parameter_segment(const TopoDS_Face& face, Vec first, Vec last, double tol_m) {
    const auto surface = BRep_Tool::Surface(face);
    const BRepAdaptor_Surface bounds(face);
    const auto project = [&](Vec p) -> std::optional<gp_Pnt2d> {
        GeomAPI_ProjectPointOnSurf result(gp_Pnt{p[0]/mm, p[1]/mm, p[2]/mm}, surface,
            bounds.FirstUParameter(), bounds.LastUParameter(), bounds.FirstVParameter(), bounds.LastVParameter(),
            Precision::PConfusion());
        if (!result.NbPoints() || result.LowerDistance()*mm > tol_m) return std::nullopt;
        double u, v;
        result.LowerDistanceParameters(u, v);
        return gp_Pnt2d{u, v};
    };
    const auto a = project(first), b = project(last);
    if (!a || !b || a->Distance(*b) <= Precision::PConfusion()) return std::nullopt;
    return FaceSegment{*a, gp_Vec2d{*a, *b}};
}

bool trim_breaks(const TopoDS_Face& face, const FaceSegment& segment, std::vector<double>& cuts) {
    const double span = segment.vector.Magnitude();
    const Handle(Geom2d_Line) axis = new Geom2d_Line(segment.start, gp_Dir2d{segment.vector});
    const Handle(Geom2d_Curve) trace = new Geom2d_TrimmedCurve(axis, 0, span);
    for (TopExp_Explorer edge(face.Oriented(TopAbs_FORWARD), TopAbs_EDGE); edge.More(); edge.Next()) {
        double first, last;
        const auto curve = BRep_Tool::CurveOnSurface(TopoDS::Edge(edge.Current()), face, first, last);
        if (curve.IsNull() || !std::isfinite(first) || !std::isfinite(last)) return false;
        if (last <= first) continue;
        const Handle(Geom2d_Curve) trim = new Geom2d_TrimmedCurve(curve, first, last);
        const Geom2dAPI_InterCurveCurve intersections(trace, trim, Precision::PConfusion());
        if (!intersections.Intersector().IsDone() || intersections.NbSegments()) return false;
        for (int i = 1; i <= intersections.NbPoints(); ++i) {
            const gp_Vec2d delta(segment.start, intersections.Point(i));
            cuts.push_back(std::clamp(delta.Dot(segment.vector)/(span*span), 0., 1.));
        }
    }
    return true;
}

std::optional<CadContactLine> straight_line(const TopoDS_Face& fa, const TopoDS_Face& fb,
    const std::vector<const CadContactCandidate*>& witnesses, double cutoff, double tolerance) {
    const CadContactCandidate* first = nullptr;
    const CadContactCandidate* last = nullptr;
    double longest = tolerance*tolerance;
    for (const auto* a : witnesses) {
        if (a->a.uv && a->b.uv) continue;
        for (const auto* b : witnesses) {
            if (b->a.uv && b->b.uv) continue;
            const double length_sq = squared(sub(b->points.a, a->points.a));
            if (length_sq > longest) {
                longest = length_sq;
                first = a;
                last = b;
            }
        }
    }
    if (!first || !last) return std::nullopt;
    const Vec start_a = first->points.a, end_a = last->points.a;
    const Vec start_b = first->points.b, end_b = last->points.b;
    const Vec direction = sub(end_a, start_a);
    const Vec direction_b = sub(end_b, start_b);
    if (length(sub(direction, direction_b)) > tolerance) return std::nullopt;
    for (const auto* witness : witnesses) {
        const double t = std::clamp(dot3(sub(witness->points.a, start_a), direction)/longest, 0., 1.);
        if (length(sub(witness->points.a, add(start_a, scale(direction, t)))) > tolerance ||
            length(sub(witness->points.b, add(start_b, scale(direction_b, t)))) > tolerance) return std::nullopt;
    }
    const auto a = parameter_segment(fa, start_a, end_a, tolerance);
    const auto b = parameter_segment(fb, start_b, end_b, tolerance);
    if (!a || !b) return std::nullopt;
    std::vector<double> cuts{0, 1};
    if (!trim_breaks(fa, *a, cuts) || !trim_breaks(fb, *b, cuts)) return std::nullopt;
    std::sort(cuts.begin(), cuts.end());
    CadContactLine line{first->a.face, first->b.face, {}};
    const BRepAdaptor_Surface sa(fa), sb(fb);
    const double full_length = std::sqrt(longest);
    for (std::size_t i = 1; i < cuts.size(); ++i) {
        const double lo = cuts[i-1], hi = cuts[i];
        const double interval_m = (hi-lo)*full_length;
        if (interval_m <= tolerance) continue;
        const double mid = std::midpoint(lo, hi);
        const auto state_a = BRepClass_FaceClassifier(fa, a->at(mid), Precision::PConfusion()).State();
        const auto state_b = BRepClass_FaceClassifier(fb, b->at(mid), Precision::PConfusion()).State();
        if (state_a == TopAbs_OUT || state_b == TopAbs_OUT) continue;
        if (state_a != TopAbs_IN || state_b != TopAbs_IN) return std::nullopt;
        for (double node : {-.57735026918962576451, .57735026918962576451}) {
            const double t = mid + (hi-lo)*.5*node;
            const gp_Pnt2d uv_a = a->at(t), uv_b = b->at(t);
            const Vec p = point(sa.Value(uv_a.X(), uv_a.Y())), q = point(sb.Value(uv_b.X(), uv_b.Y()));
            if (length(sub(p, add(start_a, scale(direction, t)))) > tolerance ||
                length(sub(q, add(start_b, scale(direction_b, t)))) > tolerance) return std::nullopt;
            const Vec delta = sub(q, p);
            const double distance = length(delta);
            if (!(distance > 1e-12 && distance <= cutoff)) return std::nullopt;
            const Vec axis = scale(delta, 1/distance);
            if (!faces_direction(fa, {line.face_a, FeatureKind::face, std::array{uv_a.X(), uv_a.Y()}}, axis) ||
                !faces_direction(fb, {line.face_b, FeatureKind::face, std::array{uv_b.X(), uv_b.Y()}}, scale(axis, -1))) {
                return std::nullopt;
            }
            line.samples.push_back({{p, q, distance}, interval_m*.5});
        }
    }
    if (line.samples.empty()) return std::nullopt;
    return line;
}

} // namespace

struct CadContactIndex::Data {
    std::vector<Face> faces;
    FaceTree tree;
};

struct CadContactCache::Data {
    std::weak_ptr<const void> a, b;
    // An empty record remembers an attempted projection that could not separate the faces.
    std::map<FacePair, std::optional<FaceExclusion>> exclusions;
};

CadContactCache::CadContactCache() : data_(std::make_unique<Data>()) {}
CadContactCache::~CadContactCache() = default;

CadContactIndex::CadContactIndex(const TopoDS_Shape& shape) {
    auto data = std::make_shared<Data>();
    TopTools_IndexedMapOfShape faces;
    TopExp::MapShapes(shape, TopAbs_FACE, faces);
    if (faces.IsEmpty()) throw std::invalid_argument("contact index requires a shape with faces");
    data->faces.reserve(faces.Extent());
    NCollection_UBTreeFiller<std::size_t, Bnd_Box> entries(data->tree);
    for (int i = 1; i <= faces.Extent(); ++i) {
        Face face{TopoDS::Face(faces(i)), {}, {}};
        BRepBndLib::AddOptimal(face.shape, face.bounds, false, true);
        if (face.bounds.IsVoid() || face.bounds.IsOpen()) {
            throw std::invalid_argument("contact index requires bounded faces");
        }
        const auto low = face.bounds.CornerMin(), high = face.bounds.CornerMax();
        face.center.SetXYZ((low.XYZ() + high.XYZ()) * .5);
        face.radius_mm = low.Distance(high) * .5;
        entries.Add(data->faces.size(), face.bounds);
        data->faces.push_back(std::move(face));
    }
    entries.Fill();
    data_ = std::move(data);
}

std::size_t CadContactIndex::face_count() const { return data_->faces.size(); }

CadContactRegions CadContactIndex::query(const CadContactIndex& b, ContactPose pose, double cutoff_m,
                                         CadContactCache* cache) const {
    const double cutoff_mm = cutoff_m / mm;
    const auto placement = relative_placement(pose, cutoff_m);
    const auto transform = placement.Transformation();
    const auto inverse = transform.Inverted();
    auto* history = cache ? cache->data_.get() : nullptr;
    if (history && (history->a.lock() != data_ || history->b.lock() != b.data_)) {
        history->exclusions.clear();
        history->a = data_;
        history->b = b.data_;
    }
    CadContactRegions result;
    for (std::size_t j = 0; j < b.face_count(); ++j) {
        const auto& local = b.data_->faces[j];
        const auto moved_bounds = local.bounds.Transformed(placement.Transformation());
        auto expanded = moved_bounds;
        // Enlarge overrides small box gaps. Keep the existing geometric margin.
        expanded.SetGap(moved_bounds.GetGap() + cutoff_mm);
        NearBox selection(expanded);
        data_->tree.Select(selection);
        if (selection.faces.empty()) continue;
        std::sort(selection.faces.begin(), selection.faces.end());
        const auto face_b = TopoDS::Face(local.shape.Moved(placement));
        Bnd_Box tight_bounds;
        BRepBndLib::AddOptimal(face_b, tight_bounds, false, true);
        BRepExtrema_DistShapeShape extrema;
        extrema.SetDeflection(1e-7);
        extrema.LoadS2(face_b);
        for (const auto i : selection.faces) {
            const auto& face_a = data_->faces[i];
            if (face_a.bounds.Distance(tight_bounds) > cutoff_mm) continue;
            if (history) {
                const auto saved = history->exclusions.find({i, j});
                if (saved != history->exclusions.end() && saved->second &&
                    stays_separated(*saved->second, local, transform, inverse, cutoff_mm)) {
                    ++result.reused_face_exclusions;
                    continue;
                }
            }
            ++result.tested_face_pairs;
            try {
                extrema.LoadS1(face_a.shape);
                if (!extrema.Perform() || !extrema.NbSolution()) {
                    ++result.unresolved_face_pairs;
                    continue;
                }
                const double distance_mm = extrema.Value();
                if (distance_mm > cutoff_mm) {
                    if (history && std::isfinite(distance_mm)) {
                        const auto [record, inserted] = history->exclusions.try_emplace(FacePair{i, j});
                        if (inserted) {
                            ++result.attempted_face_exclusions;
                            try {
                                record->second = projected_exclusion(face_a.shape, local, face_b, transform, inverse,
                                    extrema.PointOnShape1(1), extrema.PointOnShape2(1), cutoff_mm);
                            } catch (const Standard_Failure&) {
                                // Keep checking this pair with the numerical distance.
                            }
                        }
                    }
                    continue;
                }
                const std::size_t first = result.candidates.size();
                for (int k = 1; k <= extrema.NbSolution(); ++k) {
                    const Vec p = point(extrema.PointOnShape1(k)), q = point(extrema.PointOnShape2(k));
                    const Vec delta = sub(q, p);
                    const double d = length(delta);
                    if (!std::isfinite(d) || d > cutoff_m) continue;
                    auto a = face_witness(extrema, k, i + 1, true);
                    auto b_sample = face_witness(extrema, k, j + 1, false);
                    bool facing = false;
                    if (d > 1e-12 && a.uv && b_sample.uv) {
                        const Vec n = scale(delta, 1 / d);
                        facing = faces_direction(face_a.shape, a, n) && faces_direction(face_b, b_sample, scale(n, -1));
                    }
                    result.candidates.push_back({a, b_sample, {p, q, d}, facing});
                }
                if (result.candidates.size() == first) ++result.unresolved_face_pairs;
            } catch (const Standard_Failure&) {
                ++result.unresolved_face_pairs;
            }
        }
    }
    return result;
}

CadContactRegions CadContactIndex::track_interiors(const CadContactIndex& b, ContactPose pose, double cutoff_m,
                                                   std::span<const CadContactSeed> seeds) const {
    const auto placement = relative_placement(pose, cutoff_m);
    CadContactRegions result;
    result.tested_face_pairs = seeds.size();
    for (const auto& seed : seeds) {
        if (seed.face_a == 0 || seed.face_a > face_count() || seed.face_b == 0 || seed.face_b > b.face_count()) {
            throw std::invalid_argument("CAD contact seed face index is out of range");
        }
        const auto& a = data_->faces[seed.face_a - 1].shape;
        const auto moved = TopoDS::Face(b.data_->faces[seed.face_b - 1].shape.Moved(placement));
        try {
            if (auto contact = track_point(a, moved, seed, cutoff_m)) {
                result.candidates.push_back(std::move(*contact));
                continue;
            }
        } catch (const Standard_Failure&) {
            // A failed local solve leaves this seed for the face search.
        }
        ++result.unresolved_face_pairs;
    }
    return result;
}

CadContactLines CadContactIndex::straight_lines(const CadContactIndex& b, ContactPose pose, double cutoff_m,
                                                 double tolerance_m, const CadContactRegions& regions) const {
    if (!std::isfinite(tolerance_m) || tolerance_m <= 0) throw std::invalid_argument("invalid contact line tolerance");
    const auto placement = relative_placement(pose, cutoff_m);
    std::map<FacePair, std::vector<const CadContactCandidate*>> groups;
    for (const auto& candidate : regions.candidates) {
        if (candidate.facing_interiors) groups.try_emplace({candidate.a.face, candidate.b.face});
    }
    for (const auto& candidate : regions.candidates) {
        const auto found = groups.find({candidate.a.face, candidate.b.face});
        if (found != groups.end()) found->second.push_back(&candidate);
    }
    CadContactLines result;
    result.unresolved_face_pairs = regions.unresolved_face_pairs;
    for (const auto& [pair, candidates] : groups) {
        const auto [i, j] = pair;
        if (i == 0 || i > face_count() || j == 0 || j > b.face_count()) {
            throw std::invalid_argument("contact line witness face index is out of range");
        }
        try {
            const auto& a = data_->faces[i-1].shape;
            const auto moved = TopoDS::Face(b.data_->faces[j-1].shape.Moved(placement));
            if (auto line = straight_line(a, moved, candidates, cutoff_m, tolerance_m)) {
                result.lines.push_back(std::move(*line));
                continue;
            }
        } catch (const Standard_Failure&) {
            // A failed projection or trim intersection leaves the line unresolved.
        }
        ++result.unresolved_face_pairs;
    }
    return result;
}

} // namespace nc
