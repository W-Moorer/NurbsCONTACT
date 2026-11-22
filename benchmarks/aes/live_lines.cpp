// Persistent STEP contact-region oracle for the online GPU dynamics benchmark.
#include "nurbscontact/cad_contact_regions.hpp"

#include <Standard_Failure.hxx>
#include <Standard_Version.hxx>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

int main(int argc, char** argv) {
    using namespace nc;
    using namespace nc::dynamics;
    using Clock = std::chrono::steady_clock;
    try {
        const bool external=argc==5 && std::string_view(argv[4])=="--external-mask";
        if (argc!=4 && !external) throw std::invalid_argument("nc_live_lines A.step B.step cutoff_m [--external-mask]");
        const double cutoff = std::stod(argv[3]);
        if (!(cutoff>0 && std::isfinite(cutoff))) throw std::invalid_argument("invalid cutoff");
        const CadContactIndex a(read_step_shape(argv[1])), b(read_step_shape(argv[2]));
        CadContactCache cache;
        std::cout << std::setprecision(17) << "READY " << OCC_VERSION_COMPLETE << ' '
                  << a.face_count() << ' ' << b.face_count() << std::endl;
        std::vector<unsigned char> mask;
        if (external) {
            for (const auto* index : {&a,&b}) {
                for (const auto& box : index->face_bounds()) {
                    for (const auto& corner : box) for (double x : corner) std::cout << x << ' ';
                    std::cout << '\n';
                }
            }
            std::cout << std::flush;
            mask.resize(a.face_count()*b.face_count());
        }
        ContactPose pose;
        while (std::cin >> pose.origin[0]) {
            if (!(std::cin >> pose.origin[1] >> pose.origin[2] >> pose.axis[0]
                           >> pose.axis[1] >> pose.axis[2] >> pose.angle))
                throw std::invalid_argument("incomplete rigid transform");
            if (external) {
                std::size_t count,id;
                if (!(std::cin >> count) || count>mask.size()) throw std::invalid_argument("invalid mask count");
                std::fill(mask.begin(),mask.end(),0);
                for (std::size_t i=0;i<count;++i) {
                    if (!(std::cin >> id) || id>=mask.size() || mask[id]) throw std::invalid_argument("invalid mask index");
                    mask[id]=1;
                }
            }
            const auto start = Clock::now();
            const auto regions = a.query(b,pose,cutoff,&cache,mask);
            const auto assembled = Clock::now();
            const auto lines = a.straight_lines(b,pose,cutoff,1e-7,regions);
            const auto end = Clock::now();
            const auto touching = std::count_if(regions.candidates.begin(),regions.candidates.end(),
                [](const auto& candidate){ return candidate.points.distance <= 1e-12; });
            std::cout << "LINES " << lines.lines.size() << ' ' << lines.unresolved_face_pairs
                      << ' ' << regions.tested_face_pairs << ' ' << regions.reused_face_exclusions
                      << ' ' << std::chrono::duration<double>(assembled-start).count()
                      << ' ' << std::chrono::duration<double>(end-assembled).count() << ' ' << touching << '\n';
            for (const auto& line : lines.lines) {
                std::cout << line.face_a << ' ' << line.face_b << ' ' << line.samples.size() << '\n';
                for (const auto& p : line.samples) {
                    for (double x : p.points.a) std::cout << x << ' ';
                    for (double x : p.points.b) std::cout << x << ' ';
                    std::cout << p.weight_m << '\n';
                }
            }
            std::cout << std::flush;
        }
        if (!std::cin.eof()) throw std::invalid_argument("invalid pose");
    } catch (const Standard_Failure& e) {
        std::cerr << e.GetMessageString() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
