#include "nurbscontact/cpu_mesh.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

using namespace nc::cpu;

int main(int argc, char** argv) {
    try {
        if (argc < 5 || std::string_view(argv[1]) != "--method")
            throw std::invalid_argument("nc_mesh --method mesh-aabb|mesh-obb|mesh-sap [--detection discrete|continuous] [--tolerance-m metres] A.msh B.msh ...; stdin: xyz qw qx qy qz per body");
        const auto method = mesh_method(argv[2]);
        int first_mesh=3;
        bool continuous=false;
        double tolerance=1e-7;
        while (first_mesh < argc && std::string_view(argv[first_mesh]).starts_with("--")) {
            const std::string_view option=argv[first_mesh++];
            if (first_mesh==argc) throw std::invalid_argument("missing option value");
            const std::string value=argv[first_mesh++];
            if (option=="--detection") {
                if (value!="continuous" && value!="discrete") throw std::invalid_argument("unknown detection mode");
                continuous=value=="continuous";
            } else if (option=="--tolerance-m") {
                std::size_t used{};
                tolerance=std::stod(value,&used);
                if (used!=value.size() || !std::isfinite(tolerance) || tolerance<=0)
                    throw std::invalid_argument("expected a positive finite tolerance in metres");
            } else throw std::invalid_argument("unknown mesh option");
        }
        if (argc-first_mesh<2) throw std::invalid_argument("expected at least two contact meshes");
#ifndef NC_WITH_MESH_CCD
        if (continuous) throw std::invalid_argument("build with NC_WITH_MESH_CCD for continuous detection");
#endif
        std::vector<std::shared_ptr<const MeshData>> bodies;
        std::map<std::filesystem::path,std::shared_ptr<MeshData>> cached;
        for (int i = first_mesh; i < argc; ++i) {
            const auto path = std::filesystem::canonical(argv[i]);
            auto& mesh = cached[path];
            if (!mesh) mesh = std::make_shared<MeshData>(read_gmsh(path));
            bodies.push_back(mesh);
        }
        MeshScene scene(bodies, method);
        std::string line;
        std::vector<RigidPlacement> previous;
        std::cout << std::setprecision(17);
        while (std::getline(std::cin, line)) {
            std::istringstream row(line);
            std::vector<RigidPlacement> poses(bodies.size());
            for (auto& pose : poses) {
                double w, x, y, z;
                if (!(row >> pose.origin[0] >> pose.origin[1] >> pose.origin[2] >> w >> x >> y >> z) ||
                    !(std::abs(w*w+x*x+y*y+z*z-1) <= 1e-12)) throw std::runtime_error("expected a pose and unit quaternion per body");
                pose.basis = {{{1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)},
                               {2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)},
                               {2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)}}};
            }
            std::string extra;
            if (row >> extra) throw std::runtime_error("extra pose fields");
            const auto start = std::chrono::steady_clock::now();
#ifdef NC_WITH_MESH_CCD
            if (continuous) {
                if (!previous.empty()) {
                    const auto swept=scene.sweep(previous,poses,tolerance);
                    std::cout << "{\"candidate\":";
                    if (swept.first) {
                        const auto& hit=*swept.first;
                        std::cout << "{\"fraction\":" << hit.fraction << ",\"tolerance_m\":" << hit.geometric_tolerance_m
                                  << ",\"bodies\":[" << hit.body_a << ',' << hit.body_b << "],\"triangles\":["
                                  << hit.triangle_a << ',' << hit.triangle_b << "]}";
                    } else std::cout << "null";
                    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                    std::cout << ",\"body_pairs\":" << swept.body_pairs << ",\"node_pairs\":" << swept.node_pairs
                              << ",\"triangle_pairs\":" << swept.triangle_pairs << ",\"linear_tests\":" << swept.linear_tests
                              << ",\"translation_tests\":" << swept.translation_tests << ",\"query_s\":" << seconds << "}" << std::endl;
                }
                previous=std::move(poses);
                continue;
            }
#endif
            auto result = scene.intersect(poses);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            std::cout << "{\"body_pairs\":" << result.body_pairs << ",\"node_pairs\":" << result.node_pairs
                      << ",\"triangle_tests\":" << result.triangle_tests << ",\"query_s\":" << seconds << ",\"hits\":[";
            bool first = true;
            for (const auto& hit : result.intersections) {
                if (!first) std::cout << ',';
                first = false;
                std::cout << '[' << hit.body_a << ',' << hit.body_b << ',' << hit.triangle_a << ',' << hit.triangle_b << ",[";
                for (std::size_t i = 0; i < hit.points.size(); ++i) {
                    const auto& p = hit.points[i];
                    if (i) std::cout << ',';
                    std::cout << '[' << p[0] << ',' << p[1] << ',' << p[2] << ']';
                }
                std::cout << "]]";
            }
            std::cout << "],\"enclosed\":[";
            for (std::size_t i = 0; i < result.enclosed.size(); ++i) {
                const auto& item = result.enclosed[i];
                if (i) std::cout << ',';
                std::cout << '[' << item.body << ',' << item.enclosing_body << ',' << item.vertex << ']';
            }
            std::cout << "]}" << std::endl;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
