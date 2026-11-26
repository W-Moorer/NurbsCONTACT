#include "../src/compiled_geometry.hpp"
#ifdef NC_WITH_CUDA_COLUMNS
#include "../src/cuda_columns.hpp"
#endif
#include <chrono>
#include <fstream>
#include <iostream>

namespace {
using namespace nc;
using namespace nc::compiled;

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
IVec point(Vec x) {
    return {Interval{x[0]}, Interval{x[1]}, Interval{x[2]}};
}

PartFile box() {
    PartFile file;
    file.geometry.trims.emplace_back();
    Rect domain{{-1, 1}, {-1, 1}};
    for (int axis = 0; axis < 3; ++axis)
        for (int sign : {-1, 1}) {
            Vec center{}, x{}, y{}, lo{-1, -1, -1}, hi{1, 1, 1};
            center[axis] = lo[axis] = hi[axis] = sign;
            x[(axis + 1) % 3] = 1;
            y[(axis + 2) % 3] = sign;
            file.faces.push_back({domain, lo, hi, 1e-7, 1, true});
            file.geometry.patches.push_back(
                {Plane{point(center), point(x), point(y)}, domain, file.faces.size(), 0});
        }
    return file;
}
void overlap(bool cuda=false) {
    const auto file = std::filesystem::temp_directory_path() /
                      ("nc-contact-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".ncpart");
    struct Cleanup {
        std::filesystem::path file;
        ~Cleanup() {
            std::error_code e;
            std::filesystem::remove(file, e);
        }
    } cleanup{file};
    save(box(), file);
    auto a = load_solid(file), b = load_solid(file);
    auto pair = region_query(a, b, .0002, {}, cuda);
    dynamics::ContactPose pa, pb;
    dynamics::ContactWork work;
    for (double angle : {0., .4, -.7, .4}) {
        pa.angle = pb.angle = angle;
        pb.origin = pa.rotate({.0019, 0, 0});
        auto regions = pair->regions(pa, pb, work);
        require(regions.size() == 1, "one connected overlapping box region expected");
        require(std::abs(regions[0].depth_m - .0001) < 1e-11, "wrong box penetration");
        require(dot3(regions[0].normal, pa.rotate({1, 0, 0})) > 1 - 1e-10, "wrong box normal");
        require(length(sub(regions[0].point, pa.rotate({.00095, 0, 0}))) < 1e-10, "wrong pressure centroid");
        pb.origin = pa.rotate({.0021, 0, 0});
        require(pair->regions(pa, pb, work).empty(), "a finite gap produced a load");
    }
    require(!cuda || work.cuda_column_rays>0,"no CUDA columns were traced");
    std::filesystem::resize_file(file, 16);
    try {
        (void)load_solid(file);
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error("accepted a truncated part");
}
void split_and_swap() {
    auto split = box();
    const auto second = split.geometry.patches[1];
    auto right = split.faces[1];
    split.faces[1].domain.u.hi = 0;
    split.faces[1].hi[1] = 0;
    split.geometry.patches[1].domain = split.faces[1].domain;
    right.domain.u.lo = 0;
    right.lo[1] = 0;
    split.faces.push_back(right);
    split.geometry.patches.push_back({second.surface, right.domain, split.faces.size(), second.trim_id});
    auto divided = std::make_shared<Solid>(std::move(split), DiscoverySites::trim_boundary);
    auto plain = std::make_shared<Solid>(box(), DiscoverySites::trim_boundary);
    const LoadQuadrature rule{1e7, 1, 1e-6, 1e-8, 0, true};
    auto forward = region_query(divided, plain, .0002, rule);
    auto reverse = region_query(plain, divided, .0002, rule);
    dynamics::ContactPose a, b;
    dynamics::ContactWork work;
    for (double angle : {0., .2, -.4, .2}) {
        a.angle = b.angle = angle;
        b.origin = a.rotate({.0019, 0, 0});
        const auto left = forward->regions(a, b, work), right_side = reverse->regions(b, a, work);
        require(left.size() == 1 && right_side.size() == 1, "split face duplicated contact");
        auto check = [&](const dynamics::ContactRegion &r, double sign) {
            const Vec f = scale(r.normal, rule.stiffness * r.depth_m);
            const Vec moment = add(cross3(r.point, f), scale(r.pressure_arm, length(f)));
            require(length(sub(f, a.rotate({sign * 1000, 0, 0}))) < 1e-6, "split or reversed force");
            require(length(moment) < 1e-8, "split or reversed moment about shared origin");
        };
        check(left.front(), 1);
        check(right_side.front(), -1);
    }
    require(work.cad_quadrature_converged == 8, "split box missed quadrature convergence");
}
void trims() {
    PartFile file;
    Rect domain{{-2, 2}, {-2, 2}};
    TrimDomain trim;
    for (double r : {2., .5}) {
        std::array<Vec, 4> pts{{{-r, -r, 0}, {r, -r, 0}, {r, r, 0}, {-r, r, 0}}};
        trim.loops.emplace_back();
        for (int i = 0; i < 4; ++i)
            trim.loops.back().push_back(
                {BezierCurve{{homogeneous(point(pts[i])), homogeneous(point(pts[(i + 1) % 4]))}}, {}, {}, {}});
    }
    file.geometry.trims.push_back(trim);
    file.faces.push_back({domain, {-2, -2, 0}, {2, 2, 0}, 1e-7, 1, false});
    file.geometry.patches.push_back({Plane{point({}), point({1, 0, 0}), point({0, 1, 0})}, domain, 1, 0});
    Solid part(std::move(file));
    require(!part.first({.2, .1, -.1}, {0, 0, 1}, 1), "a hole generated a material exit");
    auto hit = part.first({.8, .1, -.1}, {0, 0, 1}, 1);
    require(hit && std::abs(hit->t - .1) < 1e-12, "lost the face surrounding the hole");
}
void curved_roots(bool cuda=false) {
    const Rect domain{{0, 1}, {0, 1}};
    for (bool rational : {false, true}) {
        PartFile file;
        file.geometry.trims.emplace_back();
        file.faces.push_back({domain, {0, 0, -.09}, {1, 1, .25}, 1e-7, 1, true});
        BezierSurface surface{4, 4, {}, domain};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                // Extrusion of z=(u-1/2)^2. V-dependent weights reparameterize
                // the extrusion without changing the two analytic crossings.
                const Vec pole{i / 3., j / 3., .25 - i / 3. + i * (i - 1) / 6.};
                surface.poles.push_back(homogeneous(point(pole), rational ? 1 + .1 * j : 1.));
            }
        file.geometry.patches.push_back({surface, domain, 1, 0});
        Solid solid(std::move(file));
#ifdef NC_WITH_CUDA_COLUMNS
        if(cuda) {
            auto gpu=cuda_columns(solid);
            std::vector<device::Ray> rays{
                {{{.25,.49,-.2}},{{0,0,1}}}, {{{0,.49,.01}},{{1,0,0}}},
                {{{0,.49,1e-8}},{{1,0,0}}}, {{{0,.49,-1e-6}},{{1,0,0}}},
                {{{-1,.49,-.2}},{{0,0,1}}}
            };
            std::vector<device::Hit> hits;
            gpu->trace(rays,1,hits);
            for(size_t i=0;i<hits.size();++i) {
                const auto &ray=rays[i];
                const auto ref=solid.first({ray.p.x[0],ray.p.x[1],ray.p.x[2]},
                                           {ray.d.x[0],ray.d.x[1],ray.d.x[2]},1);
                const auto &h=hits[i];
                if(i==0 && !rational) require(h.face==0,"regular cubic crossing was not resolved on device");
                if(h.face!=-2) {
                    require(h.face==(ref?ref->face:-1),"device exit face disagrees with general solver");
                    if(ref) require(std::abs(h.t-ref->t)<1e-12,"device exit depth differs");
                }
            }
        }
#endif
        for (const double z : {.01, 1e-8, -1e-6}) {
            std::vector<RayRoot> hits;
            solid.faces[0].roots({0, .49, z}, {1, 0, 0}, 1, 1e-7, 0, hits);
            if (z < 0) {
                require(hits.empty(), "near-tangent miss generated a curved root");
                continue;
            }
            std::sort(hits.begin(), hits.end(), [](const auto &a, const auto &b) { return a.t < b.t; });
            require(hits.size() == 2, "lost a near-tangent root");
            require(std::abs(hits[0].t - (.5 - std::sqrt(z))) < 2e-6 &&
                        std::abs(hits[1].t - (.5 + std::sqrt(z))) < 2e-6,
                    "incorrect curved root positions");
        }
    }
}
} // namespace
int main(int argc,char **argv) {
    try {
        bool cuda=argc==2 && std::string_view(argv[1])=="--cuda";
        require(argc==1 || cuda,"expected --cuda or no arguments");
        overlap(cuda);
        split_and_swap();
        trims();
        curved_roots(cuda);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
