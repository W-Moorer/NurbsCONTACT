#include "nurbscontact/occt.hpp"
#include "nurbscontact/query.hpp"
#include "clearance_baselines.hpp"
#include "step_triangles.hpp"

#include <BRepAdaptor_Surface.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_Copy.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRep_Builder.hxx>
#include <Bnd_Box.hxx>
#include <STEPCAFControl_Reader.hxx>
#include <TDF_LabelSequence.hxx>
#include <TDF_Tool.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <gp_Trsf.hxx>

#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <queue>
#include <span>
#include <sstream>
#include <string>
#include <string_view>

namespace {

using namespace nc;

struct Component {
    TopoDS_Shape shape;
    std::string path;
    std::string name;
};

std::string label_name(const TDF_Label& label) {
    Handle(TDataStd_Name) name;
    if (!label.FindAttribute(TDataStd_Name::GetID(), name)) return {};
    const TCollection_ExtendedString& value = name->Get();
    std::string result(static_cast<std::size_t>(value.LengthOfCString()) + 1, '\0');
    char* buffer = result.data();
    result.resize(static_cast<std::size_t>(value.ToUTF8CString(buffer)));
    return result;
}

void visit(std::vector<Component>& out, const TDF_Label& instance,
           const TopLoc_Location& parent, const std::string& path) {
    TCollection_AsciiString entry;
    TDF_Tool::Entry(instance, entry);
    const std::string next = path.empty() ? entry.ToCString() : path + "/" + entry.ToCString();
    TDF_Label definition;
    const bool reference = XCAFDoc_ShapeTool::GetReferredShape(instance, definition);
    if (!reference) definition = instance;
    TDF_LabelSequence children;
    if (XCAFDoc_ShapeTool::GetComponents(definition, children) && !children.IsEmpty()) {
        const TopLoc_Location location = parent * XCAFDoc_ShapeTool::GetLocation(instance);
        for (int i = 1; i <= children.Length(); ++i) visit(out, children(i), location, next);
        return;
    }
    const TopoDS_Shape shape = XCAFDoc_ShapeTool::GetShape(instance);
    if (!shape.IsNull() && TopExp_Explorer(shape, TopAbs_FACE).More()) {
        out.push_back({shape.Moved(parent), next, label_name(definition)});
    }
}

std::vector<Component> read_components(const char* filename) {
    STEPCAFControl_Reader reader;
    reader.SetColorMode(false);
    reader.SetLayerMode(false);
    reader.SetGDTMode(false);
    reader.SetSHUOMode(false);
    reader.SetMatMode(false);
    reader.SetPropsMode(false);
    const Handle(TDocStd_Document) document = new TDocStd_Document("BinXCAF");
    if (reader.ReadFile(filename) != IFSelect_RetDone || !reader.Transfer(document)) {
        throw std::runtime_error("cannot read assembly STEP file");
    }
    TDF_LabelSequence roots;
    XCAFDoc_DocumentTool::ShapeTool(document->Main())->GetFreeShapes(roots);
    std::vector<Component> components;
    for (int i = 1; i <= roots.Length(); ++i) visit(components, roots(i), {}, {});
    if (components.size() < 2) throw std::runtime_error("STEP file has fewer than two placed components");
    return components;
}

void csv(std::ostream& out, std::string_view value) {
    out << '"';
    for (char c : value) {
        if (c == '"') out << '"';
        out << c;
    }
    out << '"';
}

void catalog(const std::vector<Component>& components, const char* filename) {
    std::ofstream out(filename);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << std::setprecision(17) << "id,occurrence,part,faces,planes,cylinders,xmin,ymin,zmin,xmax,ymax,zmax\n";
    for (std::size_t i = 0; i < components.size(); ++i) {
        const Component& component = components[i];
        int planes = 0;
        int cylinders = 0;
        int total = 0;
        for (TopExp_Explorer faces(component.shape, TopAbs_FACE); faces.More(); faces.Next()) {
            const auto type = BRepAdaptor_Surface(TopoDS::Face(faces.Current())).GetType();
            planes += type == GeomAbs_Plane;
            cylinders += type == GeomAbs_Cylinder;
            ++total;
        }
        Bnd_Box box;
        BRepBndLib::AddOptimal(component.shape, box, false, false);
        double x0, y0, z0, x1, y1, z1;
        box.Get(x0, y0, z0, x1, y1, z1);
        out << i + 1 << ',';
        csv(out, component.path);
        out << ',';
        csv(out, component.name);
        out << ',' << total << ',' << planes << ',' << cylinders << ','
            << x0 << ',' << y0 << ',' << z0 << ',' << x1 << ',' << y1 << ',' << z1 << '\n';
    }
}

TopoDS_Shape boundary(const TopoDS_Shape& shape) {
    TopoDS_Compound result;
    BRep_Builder builder;
    builder.MakeCompound(result);
    for (TopExp_Explorer faces(shape, TopAbs_FACE); faces.More(); faces.Next()) builder.Add(result, faces.Current());
    return result;
}

double occt_distance(const TopoDS_Shape& a, const TopoDS_Shape& b, const Motion& motion, double time) {
    const Vec t = motion.translation;
    const Vec v = motion.velocity;
    gp_Trsf shift;
    shift.SetTranslation(gp_Vec{t[0] + v[0] * time, t[1] + v[1] * time, t[2] + v[2] * time});
    BRepExtrema_DistShapeShape query;
    query.SetDeflection(1e-9);
    query.LoadS1(a);
    query.LoadS2(b.Moved(TopLoc_Location{shift}));
    if (!query.Perform() || !query.IsDone()) throw std::runtime_error("OCCT distance query failed");
    return query.Value();
}

struct Reference {
    SweepStatus status{SweepStatus::unresolved};
    double lower{};
    std::optional<double> upper;
    int calls{};
};

Reference reference_check(const Component& a, const Component& b, const Motion& motion, double clearance) {
    constexpr double numerical_margin = 1e-6;
    constexpr double resolution = 2e-5;
    constexpr int max_calls = 1024;
    const TopoDS_Shape boundary_a = boundary(a.shape);
    const TopoDS_Shape boundary_b = boundary(b.shape);
    const double speed = norm(exact(motion.velocity)).hi;

    auto later = [](Interval i, Interval j) { return i.lo > j.lo; };
    std::priority_queue<Interval, std::vector<Interval>, decltype(later)> work(later);
    work.push({0, 1});
    double confirmed = infinity;
    double terminal = infinity;
    Reference result;
    while (!work.empty() && result.calls < max_calls) {
        const Interval item = work.top();
        if (item.lo >= confirmed || confirmed - std::min(item.lo, terminal) <= resolution) break;
        work.pop();
        const double middle = item.center();
        const double d = occt_distance(boundary_a, boundary_b, motion, middle);
        ++result.calls;
        // CAD-kernel oracle for cross-validation. This margin is numerical
        // protocol, not a claim of directed rounding within OCCT.
        if (d - numerical_margin - speed * item.width() / 2 > clearance) continue;
        if (d + numerical_margin <= clearance) confirmed = std::min(confirmed, middle);
        if (item.width() <= resolution) {
            terminal = std::min(terminal, item.lo);
        } else {
            work.push({item.lo, middle});
            if (middle < confirmed) work.push({middle, item.hi});
        }
    }
    result.lower = std::min({1.0, confirmed, terminal, work.empty() ? infinity : work.top().lo});
    if (std::isfinite(confirmed)) {
        result.status = SweepStatus::hit;
        result.upper = confirmed;
    } else if (work.empty() && !std::isfinite(terminal)) {
        result.status = SweepStatus::clear;
    }
    return result;
}

struct ComparisonMethod {
    const char* name;
    double deflection{};
    double prep_us{};
    std::size_t triangles{};
    std::function<nc::bench::ClearanceTiming()> run;
};

struct Scenario {
    std::string name;
    std::size_t a{};
    std::size_t b{};
    Motion motion;
    double clearance{};
};

std::vector<Scenario> read_cases(const char* filename, std::size_t parts) {
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("cannot read assembly scenarios");
    std::string line;
    std::vector<Scenario> cases;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        std::istringstream row(line);
        Scenario scene;
        Vec& t = scene.motion.translation;
        Vec& v = scene.motion.velocity;
        if (!(row >> scene.name >> scene.a >> scene.b >> t[0] >> t[1] >> t[2] >> v[0] >> v[1] >> v[2] >> scene.clearance) ||
            scene.a == 0 || scene.b == 0 || scene.a > parts || scene.b > parts || scene.a == scene.b) {
            throw std::runtime_error("invalid assembly scenario: " + line);
        }
        cases.push_back(std::move(scene));
    }
    if (cases.empty()) throw std::runtime_error("no assembly scenarios specified");
    return cases;
}

const char* status(SweepStatus value) {
    if (value == SweepStatus::clear) return "clear";
    if (value == SweepStatus::hit) return "hit";
    return "unresolved";
}

void compare(const std::vector<Component>& components, const char* config, const char* filename, int repeats,
             bool refine_mesh = false) {
    using Clock = std::chrono::steady_clock;
    using nc::bench::ClearanceTiming;
    using nc::bench::distance_advancement;
    using nc::bench::mesh_ccd;
    using nc::dynamics::TriangleBvh;
    constexpr Options settings{2e-4, 5e-4, 200'000};
    constexpr std::array standard_deflections{.05, .0125, .003125};
    constexpr std::array edge_deflections{.05, .0125, .003125, .000390625, .000048828125, .000006103515625};
    const std::span<const double> deflections = refine_mesh ? std::span<const double>(edge_deflections)
                                                            : std::span<const double>(standard_deflections);
    constexpr std::size_t max_calls = 256;
    constexpr double numerical_margin = 1e-6;
    if (repeats < 1) throw std::invalid_argument("repetition count must be positive");
    const auto elapsed = [](Clock::time_point start) {
        return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    };
    const auto cases = read_cases(config, components.size());
    std::ofstream out(filename);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << std::setprecision(17) << "scene,method,deflection_mm,repeat,clearance_mm,time_tolerance,status,"
        "lower_time,upper_time,query_us,prepare_us,calls,nodes,primitives,triangles,"
        "reference_status,reference_lower,reference_upper\n";
    for (const Scenario& scene : cases) {
        const Component& ca = components[scene.a - 1];
        const Component& cb = components[scene.b - 1];
        const auto reference = reference_check(ca, cb, scene.motion, scene.clearance);
        const double speed = norm(exact(scene.motion.velocity)).hi;

        auto start = Clock::now();
        const PreparedModel a{adapt_occt(ca.shape)};
        const PreparedModel b{adapt_occt(cb.shape)};
        const double native_prep = elapsed(start);

        start = Clock::now();
        const TopoDS_Shape fa = boundary(ca.shape);
        const TopoDS_Shape fb = boundary(cb.shape);
        BRepExtrema_DistShapeShape extrema;
        extrema.SetDeflection(1e-9);
        extrema.LoadS1(fa);
        const double occt_prep = elapsed(start);
        std::vector<ComparisonMethod> methods;
        methods.push_back({"native-sweep", 0, native_prep, 0, [&] {
            const auto answer = sweep(a, b, scene.motion, {0, 1}, scene.clearance, settings);
            return ClearanceTiming{answer.status, answer.safe_prefix, answer.witness_time, 1,
                                   answer.statistics.visited};
        }});
        const auto cad_advance = [&](std::optional<double> candidate) {
            std::size_t nodes = 0;
            const auto sample = [&](double t, double span) {
                if (nodes >= settings.max_nodes) return DistanceResult{};
                Options spatial = settings;
                spatial.distance_tolerance = std::max(settings.distance_tolerance, speed * span / 8);
                spatial.max_nodes = std::min<std::size_t>(8192, settings.max_nodes - nodes);
                DistanceResult result = distance(a, b, scene.motion, t, spatial);
                nodes += result.statistics.visited;
                return result;
            };
            return distance_advancement(sample, speed, scene.clearance, settings.time_tolerance, max_calls, candidate);
        };
        methods.push_back({"cad-ca-bounds", 0, native_prep, 0, [&] {
            return cad_advance(std::nullopt);
        }});
        methods.push_back({"occt-ca", 0, occt_prep, 0, [&] {
            const auto sample = [&](double t, double) {
                const Vec x = nc::dynamics::add(scene.motion.translation, nc::dynamics::scale(scene.motion.velocity, t));
                gp_Trsf shift;
                shift.SetTranslation(gp_Vec{x[0], x[1], x[2]});
                extrema.LoadS2(fb.Moved(TopLoc_Location{shift}));
                if (!extrema.Perform() || !extrema.IsDone()) return DistanceResult{};
                const double d = extrema.Value();
                DistanceResult result;
                result.lower = down(d - numerical_margin);
                result.upper = up(d + numerical_margin);
                return result;
            };
            return distance_advancement(sample, speed, scene.clearance, settings.time_tolerance, max_calls);
        }});

        struct MeshPair { TriangleBvh a; TriangleBvh b; };
        std::vector<MeshPair> meshes;
        meshes.reserve(deflections.size());
        const auto make_mesh = [](const TopoDS_Shape& shape, double h) {
            BRepBuilderAPI_Copy copy(shape, true, false);
            return TriangleBvh{nc::bench::step_triangles(copy.Shape(), h), 2};
        };
        for (const double h : deflections) {
            start = Clock::now();
            meshes.push_back({make_mesh(ca.shape, h), make_mesh(cb.shape, h)});
            const double mesh_prep = elapsed(start);
            const auto count = meshes.back().a.triangles().size() + meshes.back().b.triangles().size();
            const auto mesh_query = [&, index = meshes.size() - 1] {
                const auto& mesh = meshes[index];
                return mesh_ccd(mesh.a, mesh.b, scene.motion, scene.clearance, settings.time_tolerance);
            };
            methods.push_back({"mesh-bvh-ccd", h, mesh_prep, count, mesh_query});
            if (h == standard_deflections.back()) {
                methods.push_back({"mesh-ccd-cad-bounds", h, mesh_prep + native_prep, count, [&, mesh_query] {
                    const auto prediction = mesh_query();
                    auto answer = cad_advance(prediction.upper);
                    answer.calls += prediction.calls;
                    answer.primitives = prediction.primitives;
                    return answer;
                }});
            }
        }
        for (auto& method : methods) method.run();
        for (int i = 0; i < repeats; ++i) {
            for (std::size_t m = 0; m < methods.size(); ++m) {
                auto& method = methods[(m + i) % methods.size()];
                start = Clock::now();
                const auto answer = method.run();
                const double query_us = elapsed(start);
                out << scene.name << ',' << method.name << ',' << method.deflection << ',' << i << ','
                    << scene.clearance << ',' << settings.time_tolerance << ',' << status(answer.status) << ','
                    << answer.lower << ',';
                if (answer.upper) out << *answer.upper;
                out << ',' << query_us << ',' << method.prep_us << ',' << answer.calls << ',' << answer.nodes << ','
                    << answer.primitives << ',' << method.triangles << ',' << status(reference.status) << ','
                    << reference.lower << ',';
                if (reference.upper) out << *reference.upper;
                out << '\n';
            }
        }
        out.flush();
        std::cout << scene.name << " continuous baselines measured\n" << std::flush;
    }
}

void measure_clearance(
    const std::vector<Component>& components, const char* config, const char* filename
) {
    using nc::dynamics::TriangleBvh;
    using nc::dynamics::TriangleHint;
    using nc::dynamics::TriangleStatistics;
    using Clock = std::chrono::steady_clock;
    const auto cases = read_cases(config, components.size());
    std::ofstream out(filename);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << std::setprecision(17)
        << "scene,backend,deflection_mm,time,reference_mm,lower_mm,upper_mm,status,query_us,nodes,primitives,triangles\n";
    constexpr std::array times{
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.985, 0.99, 0.995, 0.999, 1.0
    };
    const auto elapsed = [](Clock::time_point start) {
        return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    };
    Options options;
    options.distance_tolerance = 2e-4;
    options.max_nodes = 30'000;
    for (const Scenario& scene : cases) {
        const TopoDS_Shape& ca = components[scene.a - 1].shape;
        const TopoDS_Shape& cb = components[scene.b - 1].shape;
        const TopoDS_Shape fa = boundary(ca);
        const TopoDS_Shape fb = boundary(cb);
        const PreparedModel a{adapt_occt(ca)};
        const PreparedModel b{adapt_occt(cb)};
        std::array<double, times.size()> oracle;
        for (std::size_t i = 0; i < times.size(); ++i) {
            oracle[i] = occt_distance(fa, fb, scene.motion, times[i]);
        }
        for (std::size_t i = 0; i < times.size(); ++i) {
            const auto start = Clock::now();
            const auto result = distance(a, b, scene.motion, times[i], options);
            const double us = elapsed(start);
            const bool resolved = std::isfinite(result.upper) &&
                result.upper - result.lower <= options.distance_tolerance;
            out << scene.name << ",brep,0," << times[i] << ',' << oracle[i] << ','
                << result.lower << ',' << result.upper << ',' << (resolved ? "ok" : "budget") << ','
                << us << ',' << result.statistics.visited << ",0,0\n";
        }
        for (const double deflection : {0.20, 0.05, 0.0125}) {
            BRepBuilderAPI_Copy copy_a(ca, true, false);
            BRepBuilderAPI_Copy copy_b(cb, true, false);
            const TriangleBvh mesh_a{nc::bench::step_triangles(copy_a.Shape(), deflection), 2};
            const TriangleBvh mesh_b{nc::bench::step_triangles(copy_b.Shape(), deflection), 2};
            const auto count = mesh_a.triangles().size() + mesh_b.triangles().size();
            TriangleHint hint;
            for (std::size_t i = 0; i < times.size(); ++i) {
                TriangleStatistics stats;
                Vec position = scene.motion.translation;
                for (std::size_t axis = 0; axis < position.size(); ++axis) {
                    position[axis] += scene.motion.velocity[axis] * times[i];
                }
                const auto start = Clock::now();
                const auto result = mesh_a.closest_within(mesh_b, nc::dynamics::Pose{position}, infinity, &hint, stats);
                const double us = elapsed(start);
                if (!result) throw std::runtime_error("triangle query has no finite witness");
                out << scene.name << ",mesh," << deflection << ',' << times[i] << ',' << oracle[i] << ",,"
                    << result->distance << ",ok," << us << ',' << stats.nodes << ',' << stats.triangles << ','
                    << count << '\n';
            }
        }
        out.flush();
        std::cout << scene.name << " clearance sampled at " << times.size() << " poses\n" << std::flush;
    }
}

void measure(const std::vector<Component>& components, const char* config, const char* filename,
             int repeats, bool all_variants) {
    if (repeats < 1) throw std::invalid_argument("repetition count must be positive");
    const auto cases = read_cases(config, components.size());
    std::ofstream out(filename);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << std::setprecision(17) << "scene,variant,repeat,status,safe_prefix,witness_time,bracket_width,query_us,"
        "nodes,spatial_splits,time_splits,axial_rejected,axial_clipped,reference_status,reference_lower,"
        "reference_upper,reference_calls,reference_agrees\n";
    bool passed = true;
    for (const Scenario& scene : cases) {
        const Component& ca = components[scene.a - 1];
        const Component& cb = components[scene.b - 1];
        const PreparedModel a{adapt_occt(ca.shape)};
        const PreparedModel b{adapt_occt(cb.shape)};
        const auto reference = reference_check(ca, cb, scene.motion, scene.clearance);
        constexpr Options base{2e-4, 5e-4, 200'000};
        std::array variants{base, base, base};
        variants[1].axial_time = variants[2].axial_time = false;
        variants[2].directional_time = false;
        constexpr std::array names{"axial", "directional", "swept-box"};
        const int methods = all_variants ? 3 : 2;
        for (int j = 0; j < methods; ++j) sweep(a, b, scene.motion, {0, 1}, scene.clearance, variants[j]);
        for (int i = 0; i < repeats; ++i) {
            for (int j = 0; j < methods; ++j) {
                const int variant = (i + j) % methods;
                const auto& options = variants[variant];
                const auto start = std::chrono::steady_clock::now();
                const auto result = sweep(a, b, scene.motion, {0, 1}, scene.clearance, options);
                const auto stop = std::chrono::steady_clock::now();
                const double us = std::chrono::duration<double, std::micro>(stop - start).count();
                const auto& stats = result.statistics;
                const bool agreed = result.status == reference.status && result.status != SweepStatus::unresolved &&
                    (result.status == SweepStatus::clear ||
                        (result.witness_time && reference.upper &&
                         result.safe_prefix <= *reference.upper && reference.lower <= *result.witness_time));
                const bool safe_unresolved = result.status == SweepStatus::unresolved &&
                    (reference.status == SweepStatus::clear ||
                     result.safe_prefix <= reference.upper.value_or(reference.lower));
                passed = passed && (all_variants ? (agreed || safe_unresolved) :
                    (agreed && (!result.witness_time || *result.witness_time - result.safe_prefix <= options.time_tolerance)));
                out << scene.name << ',' << names[variant] << ',' << i << ','
                    << status(result.status) << ',' << result.safe_prefix << ',';
                if (result.witness_time) out << *result.witness_time;
                out << ',';
                if (result.witness_time) out << *result.witness_time - result.safe_prefix;
                out << ',' << us << ',' << stats.visited << ',' << stats.spatial_splits << ',' << stats.time_splits
                    << ',' << stats.axial_rejected << ',' << stats.axial_clipped << ',' << status(reference.status)
                    << ',' << reference.lower << ',';
                if (reference.upper) out << *reference.upper;
                out << ',' << reference.calls << ',' << agreed << '\n';
            }
        }
        out.flush();
        std::cout << scene.name << " benchmarked " << ca.name << " / " << cb.name << '\n' << std::flush;
    }
    if (!passed) throw std::runtime_error("a measured scenario differed from the OCCT cross-check or time target");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string_view{argv[1]} == "catalog") {
            catalog(read_components(argv[2]), argv[3]);
        } else if (argc == 6 && (std::string_view{argv[1]} == "measure" ||
                                 std::string_view{argv[1]} == "measure-all")) {
            measure(read_components(argv[2]), argv[3], argv[4], std::stoi(argv[5]),
                    std::string_view{argv[1]} == "measure-all");
        } else if (argc == 5 && std::string_view{argv[1]} == "clearance") {
            measure_clearance(read_components(argv[2]), argv[3], argv[4]);
        } else if (argc == 6 && std::string_view{argv[1]} == "compare") {
            compare(read_components(argv[2]), argv[3], argv[4], std::stoi(argv[5]));
        } else if ((argc == 7 || (argc == 8 && std::string_view{argv[7]} == "--refine-mesh")) &&
                   std::string_view{argv[1]} == "compare-pair") {
            const std::vector<Component> parts{{read_step_shape(argv[2]), {}, {}},
                                               {read_step_shape(argv[3]), {}, {}}};
            compare(parts, argv[4], argv[5], std::stoi(argv[6]), argc == 8);
        } else {
            std::cerr << "Usage: nc_assembly_bench catalog ASSEMBLY.step OUTPUT.csv\n"
                         "       nc_assembly_bench measure ASSEMBLY.step CASES.txt OUTPUT.csv REPEATS\n"
                         "       nc_assembly_bench measure-all ASSEMBLY.step CASES.txt OUTPUT.csv REPEATS\n"
                         "       nc_assembly_bench clearance ASSEMBLY.step CASES.txt OUTPUT.csv\n"
                         "       nc_assembly_bench compare ASSEMBLY.step CASES.txt OUTPUT.csv REPEATS\n"
                         "       nc_assembly_bench compare-pair A.step B.step CASES.txt OUTPUT.csv REPEATS [--refine-mesh]\n";
            return 2;
        }
    } catch (const Standard_Failure& e) {
        std::cerr << e.GetMessageString() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
