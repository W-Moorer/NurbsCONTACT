#pragma once

#include "nurbscontact/triangle_bvh.hpp"

#include <Geom_Surface.hxx>
#include <filesystem>

namespace nc::gears {

inline constexpr double pi = 3.14159265358979323846;
inline constexpr double radians(double degrees) { return degrees * pi / 180; }

struct Gear {
    int teeth;
    double module;
    double pressure;
    double helix;
    double width;
    double backlash;
    double tip_relief{};
    double bore_radius{};
    std::vector<std::array<double,2>> measured_relief;
    double transverse_module() const;
    double transverse_pressure() const;
    double pitch_radius() const;
    double base_radius() const;
    double tip_radius() const;
    double root_radius() const;
    double twist() const;
    double roll_min() const;
    double roll_max() const;
    double bore() const;
    double half_angle(double roll) const;
    Vec flank(double u, double v) const;
};

struct QueryWork {
    std::size_t rays{};
    std::size_t evaluations{};
    std::size_t misses{};
    dynamics::TriangleStatistics mesh;
};

struct Flank {
    Flank(const std::filesystem::path& path, double deflection, unsigned stations = 0);
    std::optional<double> probe(Vec local_point, Vec local_normal, double seed, double axial,
                                unsigned station, QueryWork& work, double* slope = nullptr) const;
    Handle(Geom_Surface) face;
    std::optional<dynamics::TriangleBvh> mesh;
    double u0{}, u1{}, v0{}, v1{};

private:
    struct Span {
        double low, high, inverse_width;
        std::array<Vec,4> coefficients;
    };
    std::vector<std::vector<Span>> sections_;
};

void export_gear(const Gear& gear, const std::filesystem::path& solid,
                 const std::filesystem::path& flank, bool mirror);
double flank_fit(const Gear& gear, const Flank& flank);

}  // namespace nc::gears
