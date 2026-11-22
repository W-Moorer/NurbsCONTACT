#include "nurbscontact/cad_contact_regions.hpp"

#include <BRepAdaptor_Surface.hxx>
#include <BRepBndLib.hxx>
#include <BRepClass_FaceClassifier.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <Bnd_Box.hxx>
#include <Extrema_FuncExtSS.hxx>
#include <NCollection_UBTreeFiller.hxx>
#include <Precision.hxx>
#include <Standard_Failure.hxx>
#include <TopExp.hxx>
#include <TopLoc_Location.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <gp_Ax1.hxx>
#include <gp_Pnt2d.hxx>
#include <math_FunctionSetRoot.hxx>

#include <algorithm>
#include <stdexcept>

namespace nc {
namespace {

using namespace dynamics;
using FaceTree = NCollection_UBTree<std::size_t, Bnd_Box>;
constexpr double mm = .001;

struct Face {
    TopoDS_Face shape;
    Bnd_Box bounds;
};

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

} // namespace

struct CadContactIndex::Data {
    std::vector<Face> faces;
    FaceTree tree;
};

CadContactIndex::CadContactIndex(const TopoDS_Shape& shape) {
    auto data = std::make_shared<Data>();
    TopTools_IndexedMapOfShape faces;
    TopExp::MapShapes(shape, TopAbs_FACE, faces);
    if (faces.IsEmpty()) throw std::invalid_argument("contact index requires a shape with faces");
    data->faces.reserve(faces.Extent());
    NCollection_UBTreeFiller<std::size_t, Bnd_Box> entries(data->tree);
    for (int i = 1; i <= faces.Extent(); ++i) {
        Face face{TopoDS::Face(faces(i)), {}};
        BRepBndLib::AddOptimal(face.shape, face.bounds, false, true);
        if (face.bounds.IsVoid() || face.bounds.IsOpen()) {
            throw std::invalid_argument("contact index requires bounded faces");
        }
        entries.Add(data->faces.size(), face.bounds);
        data->faces.push_back(std::move(face));
    }
    entries.Fill();
    data_ = std::move(data);
}

std::size_t CadContactIndex::face_count() const { return data_->faces.size(); }

CadContactRegions CadContactIndex::query(const CadContactIndex& b, ContactPose pose, double cutoff_m) const {
    const double cutoff_mm = cutoff_m / mm;
    const auto placement = relative_placement(pose, cutoff_m);
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
            ++result.tested_face_pairs;
            try {
                extrema.LoadS1(face_a.shape);
                if (!extrema.Perform() || !extrema.NbSolution()) {
                    ++result.unresolved_face_pairs;
                    continue;
                }
                const double distance_mm = extrema.Value();
                if (distance_mm > cutoff_mm) continue;
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

} // namespace nc
