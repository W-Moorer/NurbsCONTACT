#include "nurbscontact/compiled_part.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    using namespace nc;
    using namespace nc::dynamics;
    try {
        const bool edges = argc == 5 && std::string_view(argv[4]) == "--audit-chart-edges";
        const bool boundary = (argc == 5 || argc == 10) && std::string_view(argv[4]) == "--trim-boundary";
        if (argc != 4 && !edges && !boundary)
            throw std::invalid_argument(
                "nc_compiled_poses A.ncpart B.ncpart maximum_depth_m [--audit-chart-edges|--trim-boundary "
                "[adaptive|fixed96 K exponent force_N moment_Nm]]");
        auto sites = edges ? compiled::DiscoverySites::audit_chart_edges : compiled::DiscoverySites::face_grid;
        if (boundary)
            sites = compiled::DiscoverySites::trim_boundary;
        std::optional<compiled::LoadQuadrature> rule;
        if (argc == 10) {
            const std::string_view mode = argv[5];
            if (mode != "fixed96" && mode != "adaptive")
                throw std::invalid_argument("unknown quadrature mode");
            rule =
                compiled::LoadQuadrature{std::stod(argv[6]), std::stod(argv[7]),         std::stod(argv[8]),
                                         std::stod(argv[9]), mode == "fixed96" ? 96 : 0, true};
        }
        auto pair = compiled::region_query(compiled::load_solid(argv[1], sites),
                                           compiled::load_solid(argv[2], sites), std::stod(argv[3]), rule);
        std::cout << std::setprecision(17);
        while ((std::cin >> std::ws).peek() != std::char_traits<char>::eof()) {
            ContactPose pose[2];
            for (auto &p : pose) {
                for (auto &x : p.origin)
                    std::cin >> x;
                for (auto &x : p.axis)
                    std::cin >> x;
                if (!(std::cin >> p.angle))
                    throw std::runtime_error("incomplete pose");
            }
            ContactWork work;
            const auto start = std::chrono::steady_clock::now();
            try {
                compiled::GroupFaces groups;
                const auto regions = pair->with_faces(pose[0], pose[1], work, groups);
                std::cout << "{\"regions\":[";
                bool first = true;
                for (auto &r : regions) {
                    if (!first)
                        std::cout << ',';
                    first = false;
                    std::cout << '[';
                    for (auto v : {r.point, r.normal, r.pressure_arm})
                        for (double x : v)
                            std::cout << x << ',';
                    std::cout << r.depth_m << ']';
                }
                std::cout << "],\"group_faces\":[";
                for (std::size_t i = 0; i < groups.size(); ++i) {
                    if (i)
                        std::cout << ',';
                    std::cout << '[';
                    for (std::size_t j = 0; j < groups[i].size(); ++j) {
                        if (j)
                            std::cout << ',';
                        auto f = groups[i][j];
                        std::cout << '[' << f[0] << ',' << f[1] << ']';
                    }
                    std::cout << ']';
                }
                std::cout << "],\"resolved\":true";
            } catch (const UnresolvedContact &e) {
                std::cout << "{\"regions\":[],\"resolved\":false";
                std::cerr << e.what() << '\n';
            }
            std::cout << ",\"query_s\":"
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                      << ",\"rules\":" << work.cad_quadrature_rules
                      << ",\"converged\":" << work.cad_quadrature_converged
                      << ",\"search_s\":" << work.cad_patch_search_s
                      << ",\"group_s\":" << work.cad_patch_group_s << ",\"area_s\":" << work.cad_patch_area_s
                      << "}\n"
                      << std::flush;
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
