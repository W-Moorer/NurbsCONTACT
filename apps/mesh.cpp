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
            throw std::invalid_argument("nc_mesh --method mesh-aabb|mesh-obb|mesh-sap [--detection discrete|continuous] [--contours|--any [--same-face-only] [--candidate-window]|--regions local|full|fcl|finite] [--tolerance-m metres] [--max-depth-m metres] A.msh B.msh ...; stdin: xyz qw qx qy qz per body");
        const auto method = mesh_method(argv[2]);
        int first_mesh=3;
        bool continuous=false;
        bool contours=false;
        bool any=false;
        bool candidate_window=false;
        bool follow_adjacent=true;
        std::optional<MeshRegionMode> region_mode;
        double tolerance=1e-7,max_depth=0.;
        while (first_mesh < argc && std::string_view(argv[first_mesh]).starts_with("--")) {
            const std::string_view option=argv[first_mesh++];
            if (option=="--contours") { contours=true; continue; }
            if (option=="--any") { any=true; continue; }
            if (option=="--candidate-window") { candidate_window=true; continue; }
            if (option=="--same-face-only") { follow_adjacent=false; continue; }
            if (first_mesh==argc) throw std::invalid_argument("missing option value");
            const std::string value=argv[first_mesh++];
            if (option=="--detection") {
                if (value!="continuous" && value!="discrete") throw std::invalid_argument("unknown detection mode");
                continuous=value=="continuous";
            } else if (option=="--regions") {
                region_mode=mesh_region_mode(value);
            } else if (option=="--max-depth-m") {
                std::size_t used{};
                max_depth=std::stod(value,&used);
                if (used!=value.size() || !std::isfinite(max_depth) || max_depth<=0) throw std::invalid_argument("invalid maximum depth");
            } else if (option=="--tolerance-m") {
                std::size_t used{};
                tolerance=std::stod(value,&used);
                if (used!=value.size() || !std::isfinite(tolerance) || tolerance<=0)
                    throw std::invalid_argument("expected a positive finite tolerance in metres");
            } else throw std::invalid_argument("unknown mesh option");
        }
        if (argc-first_mesh<2) throw std::invalid_argument("expected at least two contact meshes");
        if (continuous && contours) throw std::invalid_argument("contours require a discrete query");
        if (any && (continuous || contours)) throw std::invalid_argument("--any requires discrete overlap detection");
        if (!follow_adjacent && !any) throw std::invalid_argument("--same-face-only requires --any");
        if (candidate_window && (!any || method==MeshMethod::obb))
            throw std::invalid_argument("--candidate-window requires --any and mesh-aabb or mesh-sap");
        if (region_mode && (any || continuous || contours || argc-first_mesh!=2))
            throw std::invalid_argument("--regions takes two closed meshes for a discrete contact query");
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
        std::optional<MeshScene> scene;
        std::shared_ptr<const nc::dynamics::PairRegionQuery> regions;
        if (region_mode) regions=mesh_region_query(bodies[0],bodies[1],method,{},*region_mode,max_depth);
        else scene.emplace(bodies,method,candidate_window);
        std::string line;
        std::vector<RigidPlacement> previous;
        std::cout << std::setprecision(17);
        while (std::getline(std::cin, line)) {
            std::istringstream row(line);
            std::vector<RigidPlacement> poses(bodies.size());
            std::array<nc::dynamics::ContactPose,2> contact_poses;
            for (std::size_t i=0;i<poses.size();++i) {
                auto& pose=poses[i];
                double w, x, y, z;
                if (!(row >> pose.origin[0] >> pose.origin[1] >> pose.origin[2] >> w >> x >> y >> z) ||
                    !(std::abs(w*w+x*x+y*y+z*z-1) <= 1e-12)) throw std::runtime_error("expected a pose and unit quaternion per body");
                pose.basis = {{{1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)},
                               {2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)},
                               {2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)}}};
                if (regions) {
                    const double sine=std::hypot(x,y,z);
                    contact_poses[i]={pose.origin,sine>0 ? Vec{x/sine,y/sine,z/sine} : Vec{0,0,1},2*std::atan2(sine,w)};
                }
            }
            std::string extra;
            if (row >> extra) throw std::runtime_error("extra pose fields");
            const auto start = std::chrono::steady_clock::now();
            if (regions) {
                nc::dynamics::ContactWork work;
                const auto found=regions->regions(contact_poses[0],contact_poses[1],work);
                const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                std::cout<<"{\"query_s\":"<<seconds<<",\"triangle_tests\":"<<work.mesh.triangles<<",\"node_pairs\":"<<work.mesh.nodes
                         <<",\"boolean_queries\":"<<work.boolean_queries<<",\"input_faces\":"<<work.boolean_input_faces<<",\"local_solids\":"<<work.local_solid_uses
                         <<",\"regions\":[";
                for (std::size_t i=0;i<found.size();++i) {
                    if (i) std::cout<<',';
                    const auto& r=found[i];
                    std::cout<<"[";
                    for (double x:r.point) std::cout<<x<<',';
                    for (double x:r.normal) std::cout<<x<<',';
                    std::cout<<r.depth_m<<']';
                }
                std::cout<<"]}"<<std::endl;
                continue;
            }
#ifdef NC_WITH_MESH_CCD
            if (continuous) {
                if (!previous.empty()) {
                    const auto swept=scene->sweep(previous,poses,tolerance);
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
            if (any) {
                const auto hit=scene->overlaps(poses,follow_adjacent);
                const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                std::cout<<std::boolalpha<<"{\"overlap\":"<<hit.overlap<<",\"reused_triangle_pair\":"
                         <<hit.reused_triangle_pair<<",\"adjacent_triangle_pair\":"<<hit.adjacent_triangle_pair
                         <<",\"body_pairs\":"<<hit.body_pairs<<",\"node_pairs\":"
                         <<hit.node_pairs<<",\"triangle_tests\":"<<hit.triangle_tests<<",\"query_s\":"<<seconds<<"}"<<std::endl;
                continue;
            }
            auto result = scene->intersect(poses,contours);
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
            std::cout << ']';
            if (contours) {
                std::cout << ",\"contours\":[";
                for (std::size_t i=0;i<result.contours.size();++i) {
                    if (i) std::cout << ',';
                    const auto& group=result.contours[i];
                    std::cout << "{\"bodies\":[" << group.body_a << ',' << group.body_b << "],\"triangle_pairs\":[";
                    for (std::size_t j=0;j<group.triangles.size();++j) {
                        if (j) std::cout << ',';
                        std::cout << '[' << group.triangles[j][0] << ',' << group.triangles[j][1] << ']';
                    }
                    std::cout << "],\"vector_area_a_m2\":";
                    if (group.vector_area_a_m2) {
                        const auto& v=*group.vector_area_a_m2;
                        std::cout << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']';
                    } else std::cout << "null";
                    std::cout << '}';
                }
                std::cout << ']';
            }
            std::cout << "}" << std::endl;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
