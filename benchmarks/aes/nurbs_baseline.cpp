// Independent CPU node-to-NURBS reference for the public Motion Theory 6.3/6.4 geometry.
#include "nurbscontact/hht.hpp"
#include "nurbscontact/occt.hpp"

#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_NurbsConvert.hxx>
#include <BRepClass3d_SolidClassifier.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <GeomLib.hxx>
#include <Geom_BSplineSurface.hxx>
#include <TopExp.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Solid.hxx>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr double tol = 1e-7; // mm; distance solver tolerance in the CAD length unit
struct Node {
    gp_Pnt point;
    gp_Dir normal;
};
struct Projection {
    gp_Pnt point;
    gp_Dir normal;
    double gap, tangential;
    int face, normal_order;
    gp_Vec du, dv;
};
struct Bounds {
    std::array<double, 3> lo, hi;
    double depth_bound(const gp_Pnt& point) const {
        double upper = std::numeric_limits<double>::infinity();
        for (int i = 0; i < 3; ++i)
            upper = std::min({upper, point.Coord(i + 1) - lo[i], hi[i] - point.Coord(i + 1)});
        return upper;
    }
};
Bounds cad_bounds(const TopoDS_Shape& shape) {
    Bnd_Box box;
    BRepBndLib::AddOptimal(shape, box, false, true);
    Bounds result;
    box.Get(result.lo[0], result.lo[1], result.lo[2], result.hi[0], result.hi[1], result.hi[2]);
    return result;
}

class Body {
  public:
    explicit Body(const char* path, double deflection) {
        BRepBuilderAPI_NurbsConvert convert(nc::read_step_shape(path), true);
        TopTools_IndexedMapOfShape solids;
        TopExp::MapShapes(convert.Shape(), TopAbs_SOLID, solids);
        if (solids.Extent() != 1)
            throw std::runtime_error("Expected one closed STEP solid");
        solid_ = TopoDS::Solid(solids(1));
        TopExp::MapShapes(solid_, TopAbs_FACE, faces_);
        TopExp::MapShapesAndAncestors(solid_, TopAbs_EDGE, TopAbs_FACE, edge_faces_);
        TopExp::MapShapesAndAncestors(solid_, TopAbs_VERTEX, TopAbs_FACE, vertex_faces_);
        local_bounds = cad_bounds(solid_);
        classifier_.Load(solid_);
        BRep_Builder builder;
        TopoDS_Compound boundary;
        builder.MakeCompound(boundary); // Distance to the shell stays nonzero for interior nodes.
        if (deflection > 0) {
            BRepMesh_IncrementalMesh mesh(solid_, deflection, false, 0.5, false);
            if (!mesh.IsDone())
                throw std::runtime_error("Cannot seed nodes on the STEP faces");
        }
        for (int i = 1; i <= faces_.Extent(); ++i) {
            const auto face = TopoDS::Face(faces_(i));
            auto surface = Handle(Geom_BSplineSurface)::DownCast(BRep_Tool::Surface(face));
            if (surface.IsNull())
                throw std::runtime_error("Expected a converted NURBS face");
            surfaces_.push_back(surface);
            builder.Add(boundary, face);
            if (deflection == 0)
                continue;
            TopLoc_Location loc;
            const auto mesh = BRep_Tool::Triangulation(face, loc);
            if (mesh.IsNull() || !mesh->HasUVNodes())
                throw std::runtime_error("Missing CAD node parameters");
            for (int j = 1; j <= mesh->NbNodes(); ++j) {
                const auto uv = mesh->UVNode(j);
                gp_Dir normal;
                if (normal_at(i, uv, normal) >= 2)
                    throw std::runtime_error("Undefined source normal");
                nodes.push_back({surface->Value(uv.X(), uv.Y()), normal});
            }
        }
        nearest_.SetDeflection(tol);
        nearest_.LoadS2(boundary);
    }

    Projection project(const gp_Pnt& point) {
        ++queries;
        nearest_.LoadS1(BRepBuilderAPI_MakeVertex(point).Vertex());
        if (!nearest_.Perform() || !nearest_.NbSolution())
            throw std::runtime_error("CAD closest point failed");
        // A seam or pole can be returned as a topological edge or vertex on a smooth surface.
        // Only accept it as a node-to-face contact if the connector satisfies the same normal equation.
        for (int j = 1; j <= nearest_.NbSolution(); ++j) {
            const auto support = nearest_.SupportOnShape2(j);
            const auto kind = nearest_.SupportTypeShape2(j);
            TopTools_ListOfShape incident;
            if (kind == BRepExtrema_IsInFace)
                incident.Append(support);
            else
                incident = (kind == BRepExtrema_IsOnEdge ? edge_faces_ : vertex_faces_).FindFromKey(support);
            for (const auto& item : incident) {
                int id = faces_.FindIndex(item);
                const auto face = TopoDS::Face(faces_(id));
                gp_Pnt2d uv;
                if (kind == BRepExtrema_IsInFace) {
                    double u, v;
                    nearest_.ParOnFaceS2(j, u, v);
                    uv.SetCoord(u, v);
                } else if (kind == BRepExtrema_IsVertex) {
                    uv = BRep_Tool::Parameters(TopoDS::Vertex(support), face);
                } else {
                    double t, first, last;
                    nearest_.ParOnEdgeS2(j, t);
                    auto curve = BRep_Tool::CurveOnSurface(TopoDS::Edge(support), face, first, last);
                    if (curve.IsNull())
                        continue;
                    uv = curve->Value(t);
                }
                gp_Pnt q;
                gp_Vec du, dv;
                surfaces_[id - 1]->D1(uv.X(), uv.Y(), q, du, dv);
                gp_Dir normal;
                int order = normal_at(id, uv, normal);
                if (order >= 2 || q.Distance(nearest_.PointOnShape2(j)) > 10 * tol)
                    continue;
                const gp_Vec delta(q, point);
                const double gap = delta.Dot(gp_Vec(normal));
                const double tangent = delta.Crossed(gp_Vec(normal)).Magnitude();
                if (tangent <= 10 * tol + 1e-8 * delta.Magnitude())
                    return {q, normal, gap, tangent, id, order + 1, du, dv};
            }
        }
        // A sharp feature with no admissible face normal is explicitly distinguishable
        // from an open gap. Keep its signed distance so the force caller can reject it.
        classifier_.Perform(point, tol);
        const auto side = classifier_.State();
        if (side == TopAbs_UNKNOWN)
            throw std::runtime_error("CAD could not classify this node");
        const double gap = (side == TopAbs_IN ? -1 : side == TopAbs_OUT ? 1 : 0) * nearest_.Value();
        return {nearest_.PointOnShape2(1), gp_Dir(1, 0, 0), gap, 0, 0, 0, {}, {}};
    }
    const Bounds* fixed_orientation_bounds(gp_Trsf pose) {
        std::array<double, 9> rotation;
        for (int i = 0; i < 9; ++i)
            rotation[i] = pose.Value(1 + i / 3, 1 + i % 3);
        if (rotation != last_rotation_) {
            last_rotation_ = rotation;
            fixed_bounds_.reset();
            return nullptr;
        }
        // Repeated orientations also permit a tight world-axis box, reused under translation.
        if (!fixed_bounds_) {
            pose.SetTranslationPart(gp_Vec{});
            fixed_bounds_ = cad_bounds(solid_.Moved(TopLoc_Location(pose)));
        }
        return &*fixed_bounds_;
    }
    int face_count() const { return faces_.Extent(); }
    bool inside(const gp_Pnt& p) {
        ++classifications;
        classifier_.Perform(p, tol);
        if (classifier_.State() == TopAbs_UNKNOWN)
            throw std::runtime_error("Undefined node classification");
        return classifier_.State() == TopAbs_IN;
    }
    std::vector<Node> nodes;
    Bounds local_bounds;
    std::size_t queries{}, classifications{};

  private:
    int normal_at(int face, const gp_Pnt2d& uv, gp_Dir& normal) const {
        int status = GeomLib::NormEstim(surfaces_[face - 1], uv, 1e-10, normal);
        if (faces_(face).Orientation() == TopAbs_REVERSED)
            normal.Reverse();
        return status;
    }
    TopoDS_Solid solid_;
    TopTools_IndexedMapOfShape faces_;
    TopTools_IndexedDataMapOfShapeListOfShape edge_faces_, vertex_faces_;
    std::vector<Handle(Geom_BSplineSurface)> surfaces_;
    std::array<double, 9> last_rotation_{};
    std::optional<Bounds> fixed_bounds_;
    BRepClass3d_SolidClassifier classifier_;
    BRepExtrema_DistShapeShape nearest_;
};

struct Contact {
    double depth{};
    gp_Dir normal{0, 0, 1}; // from body A toward B, in world coordinates
    gp_Pnt on_a, on_b;
};
// One linear spring per body pair, as in the repository's concentrated-stiffness control.
Contact pair_contact(Body& a, const gp_Trsf& pa, Body& b, const gp_Trsf& pb) {
    Contact best;
    double unsupported = 0;
    for (bool reverse : {false, true}) {
        auto& target = reverse ? b : a;
        const auto& source = reverse ? a : b;
        const auto& pt = reverse ? pb : pa;
        const auto& ps = reverse ? pa : pb;
        const auto rel = pt.Inverted() * ps;
        const auto extra = target.fixed_orientation_bounds(pt);
        std::vector<std::pair<double, std::size_t>> candidates;
        for (std::size_t i = 0; i < source.nodes.size(); ++i) {
            const auto& point = source.nodes[i].point;
            double bound = target.local_bounds.depth_bound(point.Transformed(rel));
            if (extra && bound > best.depth)
                bound = std::min(bound, extra->depth_bound(
                                            point.Transformed(ps).Translated(-gp_Vec(pt.TranslationPart()))));
            if (bound > best.depth)
                candidates.emplace_back(bound, i);
        }
        std::sort(candidates.rbegin(), candidates.rend());
        for (auto [upper, i] : candidates) {
            // For an interior point, a ray to any outer AABB plane crosses the boundary first.
            if (upper <= best.depth)
                break;
            const auto node = source.nodes[i].point.Transformed(rel);
            if (!target.inside(node))
                continue;
            const auto hit = target.project(node);
            if (-hit.gap <= best.depth)
                continue;
            if (!hit.face) {
                unsupported = std::max(unsupported, -hit.gap);
                continue;
            }
            if (hit.normal.Dot(source.nodes[i].normal.Transformed(rel)) >= 0)
                continue;
            best.depth = -hit.gap;
            best.normal = hit.normal.Transformed(pt);
            best.on_a = (reverse ? node : hit.point).Transformed(pt);
            best.on_b = (reverse ? hit.point : node).Transformed(pt);
            if (reverse)
                best.normal.Reverse();
        }
    }
    if (unsupported > best.depth + tol)
        throw std::runtime_error("Governing sharp feature needs an edge contact law");
    return best;
}

gp_Trsf pose(const gp_Vec& position, const gp_Dir& axis, double angle) {
    gp_Trsf transform;
    transform.SetRotation(gp_Ax1(gp::Origin(), axis), angle);
    transform.SetTranslationPart(position);
    return transform;
}
void sliders(Body& a, Body& b, double tilt, double h, std::ostream& out) {
    // Existing AES equal-mass impact fixture. Only the driver knows the prescribed joint axes.
    using namespace nc::dynamics;
    const gp_Dir axis(0, 0, 1), parameter_rotation(1, 2, 0.5);
    auto forces = [&](const Vector& q, const Vector&, double) {
        auto hit = pair_contact(a, pose(gp_Vec(0, 0, 1000 * q[0]), parameter_rotation, tilt), b,
                                pose(gp_Vec(0, 0, 140 + 1000 * q[1]), parameter_rotation, -0.7 * tilt));
        double nz = hit.normal.Dot(axis);
        double f = 5 * hit.depth * nz; // 5000 N/m, geometry in mm
        double k = hit.depth > 0 ? 5000 * nz * nz : 0;
        return Forces{{-f, f}, {k, -k, -k, k}, {0, 0, 0, 0}};
    };
    Hht solver({1, 1}, forces);
    auto state = solver.initialize({0, 0}, {0.225, -0.225});
    HhtStats stats;
    int count = static_cast<int>(std::round(0.16 / h));
    if (count < 1 || std::abs(count * h - 0.16) > 1e-12)
        throw std::runtime_error("Step must divide 0.16 s");
    out << "time_s,q0_m,q1_m,v0_m_s,v1_m_s,axial_force_N\n";
    for (int i = 0; i <= count; ++i) {
        if (i)
            solver.step(state, h, stats);
        out << state.time << ',' << state.q[0] << ',' << state.q[1] << ',' << state.v[0] << ',' << state.v[1]
            << ',' << state.force[1] << '\n';
    }
    std::cerr << "hht_evaluations=" << stats.evaluations << " subdivisions=" << stats.subdivisions << ' ';
}

int main_impl(int argc, char** argv) {
    if (argc < 4)
        throw std::runtime_error("points TARGET.step output.csv < xyz_mm.txt | pair A.step B.step "
                                 "deflection_mm output.csv < poses.txt"
                                 " | impact A.step B.step deflection_mm tilt_rad step_s output.csv");
    const std::string mode = argv[1];
    if (!(mode == "points" && argc == 4) && !(mode == "pair" && argc == 6) &&
        !(mode == "impact" && argc == 8))
        throw std::runtime_error("Invalid baseline arguments");
    const double deflection = mode == "points" ? 0 : std::stod(argv[4]);
    if (mode != "points" && !(deflection > 0 && std::isfinite(deflection)))
        throw std::runtime_error("Invalid deflection");
    const auto start = Clock::now();
    Body a(argv[2], deflection);
    std::unique_ptr<Body> b;
    if (mode != "points")
        b = std::make_unique<Body>(argv[3], deflection);
    std::ofstream out(argv[argc - 1]);
    if (!out)
        throw std::runtime_error("Cannot open output");
    out << std::setprecision(17);
    const auto ready = Clock::now();
    if (mode == "points") {
        out << "gap_mm,qx_mm,qy_mm,qz_mm,nx,ny,nz,tangential_mm,face,normal_derivative_order,"
               "dux,duy,duz,dvx,dvy,dvz\n";
        double x, y, z;
        while (std::cin >> x) {
            if (!(std::cin >> y >> z))
                throw std::runtime_error("Incomplete input node");
            auto p = a.project(gp_Pnt(x, y, z));
            out << p.gap << ',' << p.point.X() << ',' << p.point.Y() << ',' << p.point.Z() << ',';
            if (p.face)
                out << p.normal.X() << ',' << p.normal.Y() << ',' << p.normal.Z() << ',' << p.tangential;
            else
                out << "nan,nan,nan,nan";
            out << ',' << p.face << ',' << p.normal_order;
            if (p.face)
                out << ',' << p.du.X() << ',' << p.du.Y() << ',' << p.du.Z() << ',' << p.dv.X() << ','
                    << p.dv.Y() << ',' << p.dv.Z();
            else
                out << ",nan,nan,nan,nan,nan,nan";
            out << '\n';
        }
    } else if (mode == "impact") {
        double h = std::stod(argv[6]);
        if (!(h > 0 && h <= 0.16 && std::isfinite(h)))
            throw std::runtime_error("Invalid impact time step");
        sliders(a, *b, std::stod(argv[5]), h, out);
    } else {
        out << "depth_mm,nx,ny,nz,ax_mm,ay_mm,az_mm,bx_mm,by_mm,bz_mm\n";
        while (std::cin >> std::ws && std::cin.peek() != EOF) {
            gp_Trsf transforms[2];
            for (auto& transform : transforms) {
                double x, y, z, u, v, w, angle;
                if (!(std::cin >> x >> y >> z >> u >> v >> w >> angle))
                    throw std::runtime_error("Expected two poses");
                transform = pose(gp_Vec(x, y, z), gp_Dir(u, v, w), angle);
            }
            auto c = pair_contact(a, transforms[0], *b, transforms[1]);
            out << c.depth;
            if (c.depth > 0)
                out << ',' << c.normal.X() << ',' << c.normal.Y() << ',' << c.normal.Z() << ',' << c.on_a.X()
                    << ',' << c.on_a.Y() << ',' << c.on_a.Z() << ',' << c.on_b.X() << ',' << c.on_b.Y() << ','
                    << c.on_b.Z();
            else
                out << ",nan,nan,nan,nan,nan,nan,nan,nan,nan";
            out << '\n';
        }
    }
    if (std::cin.fail() && !std::cin.eof())
        throw std::runtime_error("Invalid input");
    out.close();
    if (!out)
        throw std::runtime_error("Cannot write baseline output");
    std::cerr << std::setprecision(10) << "setup_s=" << std::chrono::duration<double>(ready - start).count()
              << " online_s=" << std::chrono::duration<double>(Clock::now() - ready).count()
              << " queries=" << a.queries + (b ? b->queries : 0)
              << " classifications=" << a.classifications + (b ? b->classifications : 0)
              << " faces_a=" << a.face_count() << " nodes_a=" << a.nodes.size()
              << " nodes_b=" << (b ? b->nodes.size() : 0) << '\n';
    return 0;
}
} // namespace
int main(int argc, char** argv) {
    try {
        return main_impl(argc, argv);
    } catch (const Standard_Failure& e) {
        std::cerr << e.GetMessageString() << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    return 1;
}
