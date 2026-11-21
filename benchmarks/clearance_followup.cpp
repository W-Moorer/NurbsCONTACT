#include "nurbscontact/occt.hpp"
#include "nurbscontact/query.hpp"

#include <BRepAlgoAPI_Cut.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <STEPControl_Writer.hxx>
#include <gp_Circ.hxx>
#include <gp_Pln.hxx>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {
using namespace nc;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

constexpr double clearance = .03125;
constexpr Options settings{2e-4, 5e-4, 200'000};

TopoDS_Shape cylinder(double radius, double bottom, double height) {
    const gp_Ax2 axis{gp_Pnt{0, 0, bottom}, gp_Dir{0, 0, 1}};
    return BRepPrimAPI_MakeCylinder(axis, radius, height).Shape();
}

TopoDS_Wire circle(double radius) {
    // Put the seam away from the independently closest point at +x.
    const gp_Ax2 axis{gp_Pnt{0, 0, 0}, gp_Dir{0, 0, 1}, gp_Dir{0, 1, 0}};
    return BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(gp_Circ{axis, radius}).Edge()).Wire();
}

PreparedModel step_model(const TopoDS_Shape& shape, const fs::path& output) {
    STEPControl_Writer writer;
    if (writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone ||
        writer.Write(output.string().c_str()) != IFSelect_RetDone) {
        throw std::runtime_error("could not write " + output.string());
    }
    return PreparedModel(read_step(output));
}

const char* status(SweepStatus status) {
    if (status == SweepStatus::hit) return "hit";
    return status == SweepStatus::clear ? "clear" : "unresolved";
}

const char* feature(FeatureKind kind) {
    if (kind == FeatureKind::edge) return "edge";
    return kind == FeatureKind::vertex ? "vertex" : "face";
}

void sample(std::ostream& out, const Sample& point) {
    out << ',' << feature(point.feature.kind) << ',' << point.feature.id;
    for (Interval x : point.position) out << ',' << x.lo << ',' << x.hi;
    for (Interval x : point.uv) out << ',' << x.lo << ',' << x.hi;
}

void witness(std::ostream& out, const std::optional<Witness>& points) {
    if (points) {
        sample(out, points->a);
        sample(out, points->b);
    } else {
        for (int i = 0; i < 24; ++i) out << ',';
    }
}

struct Scenario {
    const char* name;
    double offset, z, speed;
};

void measure(std::ostream& out, const PreparedModel& a, const PreparedModel& b,
             Scenario scene, int count, bool audit) {
    std::array<Options, 3> variants{settings, settings, settings};
    variants[0].directional_time = false;
    variants[0].axial_time = variants[1].axial_time = false;
    constexpr std::array names{"swept-box", "directional", "axial"};
    const Motion motion{{scene.offset, 0, scene.z}, {0, 0, scene.speed}};
    const auto query = [&](std::size_t variant) {
        return sweep(a, b, motion, {0, 1}, clearance, variants[variant]);
    };
    const std::size_t methods = audit ? 1 : variants.size();
    for (std::size_t j = 0; j < methods; ++j) query(audit ? 2 : j);
    for (int i = 0; i < count; ++i) {
        for (std::size_t j = 0; j < methods; ++j) {
            const std::size_t variant = audit ? 2 : (i+j)%methods;
            const auto start = Clock::now();
            const auto result = query(variant);
            const auto us = std::chrono::duration<double, std::micro>(Clock::now()-start).count();
            const auto& work = result.statistics;
            out << scene.name << ',' << names[variant] << ',' << i << ',' << scene.offset << ','
                << scene.z << ',' << scene.speed << ',' << status(result.status) << ','
                << result.safe_prefix << ',' << result.witness_time.value_or(nc::infinity) << ',' << us << ','
                << work.visited << ',' << work.spatial_splits << ',' << work.time_splits << ','
                << work.axial_rejected << ',' << work.axial_clipped;
            witness(out, result.witness);
            out << '\n';
        }
    }
    out.flush();
    std::cout << "Measured " << scene.name << '\n' << std::flush;
}

void counterbore(std::ostream& out, const fs::path& root, int runs) {
    const auto blank = cylinder(2.5, -3, 5);
    const auto through = BRepAlgoAPI_Cut(blank, cylinder(.8125, -4, 7)).Shape();
    const auto cut = BRepAlgoAPI_Cut(through, cylinder(1.375, 0, 3)).Shape();
    const auto a = step_model(cut, root/"counterbore.step");
    const auto b = step_model(cylinder(.375, 0, .5), root/"pin.step");
    for (const Scenario scene : {Scenario{"throat-pass", .375, 3, -4},
                                 Scenario{"shoulder-entry", .4140625, 3, -4},
                                 Scenario{"mouth-entry", .9765625, 3, -4}}) {
        measure(out, a, b, scene, runs, false);
    }
}

void audit_edges(std::ostream& out, const fs::path& root, int runs) {
    const gp_Pln plane{gp_Pnt{0, 0, 0}, gp_Dir{0, 0, 1}};
    auto hole = circle(1);
    hole.Reverse();
    BRepBuilderAPI_MakeFace plate(plane, -3, 3, -3, 3);
    plate.Add(hole);
    const auto a = step_model(plate.Shape(), root/"edge-plate.step");
    const auto b = step_model(BRepBuilderAPI_MakeFace(plane, circle(.25)).Shape(), root/"edge-disk.step");
    for (const Scenario scene : {Scenario{"trim-edge-entry", .75-.0234375, .375, -1},
                                 Scenario{"narrow-edge-entry", .75-clearance+0x1p-26, .375, -1},
                                 Scenario{"exact-edge-graze", .75-clearance, .375, -1},
                                 Scenario{"near-edge-miss", .75-clearance-0x1p-23, .375, -1}}) {
        measure(out, a, b, scene, runs, true);
    }
}
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: nc_clearance_followup counterbore|audit STEP_DIR OUTPUT.csv REPEATS\n";
        return 2;
    }
    try {
        const std::string_view mode = argv[1];
        const int rounds = std::stoi(argv[4]);
        if (rounds < 1 || (mode != "counterbore" && mode != "audit")) {
            throw std::invalid_argument("invalid experiment mode or repetition count");
        }
        fs::create_directories(argv[2]);
        std::ofstream out(argv[3]);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out << std::setprecision(17)
            << "scene,method,repeat,offset,z_start,z_rate,status,safe_prefix,witness_time,query_us,"
            << "nodes,spatial_splits,time_splits,axial_rejected,axial_clipped";
        for (const char body : {'a', 'b'}) {
            out << ',' << body << "_kind," << body << "_id";
            for (char axis : {'x', 'y', 'z', 'u', 'v'}) {
                out << ',' << body << '_' << axis << "_lo," << body << '_' << axis << "_hi";
            }
        }
        out << '\n';
        if (mode == "counterbore") counterbore(out, argv[2], rounds);
        else audit_edges(out, argv[2], rounds);
    } catch (const Standard_Failure& error) {
        std::cerr << error.GetMessageString() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
