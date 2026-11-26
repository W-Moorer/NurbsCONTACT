#include "nurbscontact/compiled_part.hpp"
#include <bit>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <limits>

namespace nc::compiled {
namespace {
static_assert(std::endian::native == std::endian::little && sizeof(double) == 8);
using Size = std::uint32_t;
constexpr std::uint64_t signature = 0x315452415044434eull; // NCDPART1
struct Writer {
    std::ostream &file;
    template <class T> void value(const T &x) {
        static_assert(std::is_trivially_copyable_v<T>);
        file.write(reinterpret_cast<const char *>(&x), sizeof(T));
    }
    void count(std::size_t n) {
        if (n > std::numeric_limits<Size>::max())
            throw std::length_error("CAD table too large");
        value(static_cast<Size>(n));
    }
};
struct Reader {
    std::istream &file;
    std::uintmax_t remaining;
    template <class T> T value() {
        static_assert(std::is_trivially_copyable_v<T>);
        if (remaining < sizeof(T))
            throw std::runtime_error("truncated CAD part");
        T x{};
        file.read(reinterpret_cast<char *>(&x), sizeof(T));
        remaining -= sizeof(T);
        return x;
    }
    Size count(std::size_t min_bytes = 1) {
        auto n = value<Size>();
        if (n > remaining / min_bytes)
            throw std::runtime_error("invalid CAD table size");
        return n;
    }
};
void feature(Writer &w, Feature f) {
    w.value(static_cast<Size>(f.kind));
    w.count(f.id);
}
Feature feature(Reader &r) {
    const auto k = r.value<Size>();
    if (k > 2)
        throw std::runtime_error("invalid CAD feature");
    return {static_cast<FeatureKind>(k), r.value<Size>()};
}
void poles(Writer &w, const std::vector<HPoint> &p) {
    w.count(p.size());
    for (auto &h : p)
        w.value(h);
}
std::vector<HPoint> poles(Reader &r) {
    std::vector<HPoint> p(r.count(sizeof(HPoint)));
    if (p.size() < 2)
        throw std::runtime_error("empty or constant CAD control polygon");
    for (auto &h : p) {
        h = r.value<HPoint>();
        if (!(h[3].lo > 0) || std::any_of(h.begin(), h.end(), [](Interval a) {
                return !(std::isfinite(a.lo) && std::isfinite(a.hi) && a.lo <= a.hi);
            }))
            throw std::runtime_error("invalid CAD weight");
    }
    return p;
}
} // namespace
void save(const PartFile &part, const std::filesystem::path &path) {
    auto temporary = path;
    temporary += "." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".tmp";
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    } cleanup{temporary};
    std::ofstream out(temporary, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    Writer w{out};
    w.value(signature);
    w.count(part.faces.size());
    for (const auto &f : part.faces) {
        w.value(f.domain);
        w.value(f.lo);
        w.value(f.hi);
        w.value(f.tolerance);
        w.value(f.orientation);
        w.value(static_cast<Size>(f.rectangle));
    }
    w.count(part.geometry.trims.size());
    for (const auto &t : part.geometry.trims) {
        w.count(t.loops.size());
        for (const auto &loop : t.loops) {
            w.count(loop.size());
            for (const auto &c : loop) {
                feature(w, c.feature);
                feature(w, c.start);
                feature(w, c.end);
                w.count(c.geometry.index());
                if (auto b = std::get_if<BezierCurve>(&c.geometry))
                    poles(w, b->poles);
                else {
                    const auto &a = std::get<Arc>(c.geometry);
                    w.value(a.center);
                    w.value(a.x);
                    w.value(a.y);
                    w.value(a.angles);
                    w.value(static_cast<Size>(a.reversed));
                }
            }
        }
    }
    w.count(part.geometry.patches.size());
    for (const auto &p : part.geometry.patches) {
        w.count(p.face_id);
        w.count(p.trim_id);
        w.value(p.domain);
        w.count(p.surface.index());
        std::visit(
            [&](const auto &s) {
                if constexpr (std::is_same_v<std::decay_t<decltype(s)>, BezierSurface>) {
                    w.count(s.nu);
                    w.count(s.nv);
                    w.value(s.domain);
                    poles(w, s.poles);
                } else
                    w.value(s);
            },
            p.surface);
    }
    out.close();
    std::filesystem::rename(temporary, path);
}
PartFile load(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open compiled CAD part: " + path.string());
    in.exceptions(std::ios::failbit | std::ios::badbit);
    Reader r{in, std::filesystem::file_size(path)};
    if (r.value<std::uint64_t>() != signature)
        throw std::runtime_error("unknown compiled CAD part format");
    PartFile part;
    part.faces.resize(r.count(sizeof(Rect) + sizeof(Vec) * 2 + sizeof(double) + 2 * sizeof(Size)));
    for (auto &f : part.faces) {
        f.domain = r.value<Rect>();
        f.lo = r.value<Vec>();
        f.hi = r.value<Vec>();
        f.tolerance = r.value<double>();
        f.orientation = r.value<int>();
        f.rectangle = r.value<Size>() != 0;
        if ((f.orientation != 1 && f.orientation != -1) || !(std::isfinite(f.tolerance) && f.tolerance >= 0))
            throw std::runtime_error("invalid CAD face orientation or tolerance");
        for (int k = 0; k < 3; ++k)
            if (!(std::isfinite(f.lo[k]) && std::isfinite(f.hi[k]) && f.lo[k] <= f.hi[k]))
                throw std::runtime_error("invalid CAD face bounds");
        for (Interval v : {f.domain.u, f.domain.v})
            if (!(std::isfinite(v.width()) && v.width() > 0))
                throw std::runtime_error("invalid CAD parameter domain");
    }
    auto &model = part.geometry;
    model.trims.resize(r.count(sizeof(Size)));
    for (auto &t : model.trims) {
        t.loops.resize(r.count(sizeof(Size)));
        for (auto &loop : t.loops) {
            loop.resize(r.count(7 * sizeof(Size)));
            for (auto &c : loop) {
                c.feature = feature(r);
                c.start = feature(r);
                c.end = feature(r);
                switch (r.value<Size>()) {
                case 0:
                    c.geometry = BezierCurve{poles(r)};
                    break;
                case 1:
                    c.geometry = Arc{r.value<IVec>(), r.value<IVec>(), r.value<IVec>(), r.value<Interval>(),
                                     r.value<Size>() != 0};
                    break;
                default:
                    throw std::runtime_error("unknown CAD trim curve");
                }
            }
        }
    }
    model.patches.resize(r.count(sizeof(Rect) + 3 * sizeof(Size)));
    for (auto &p : model.patches) {
        p.face_id = r.value<Size>();
        p.trim_id = r.value<Size>();
        p.domain = r.value<Rect>();
        if (!p.face_id || p.face_id > part.faces.size() || p.trim_id >= model.trims.size())
            throw std::runtime_error("invalid CAD face reference");
        switch (r.value<Size>()) {
        case 0:
            p.surface = r.value<Plane>();
            break;
        case 1:
            p.surface = r.value<Cylinder>();
            break;
        case 2:
            p.surface = r.value<Revolved>();
            break;
        case 3: {
            BezierSurface s{r.value<Size>(), r.value<Size>(), {}, r.value<Rect>()};
            s.poles = poles(r);
            if (s.nu < 2 || s.nv < 2 || s.nu * s.nv != s.poles.size())
                throw std::runtime_error("invalid CAD control net");
            p.surface = std::move(s);
            break;
        }
        default:
            throw std::runtime_error("unknown CAD surface");
        }
    }
    if (r.remaining || model.patches.empty())
        throw std::runtime_error("invalid compiled part length");
    return part;
}
} // namespace nc::compiled
