#pragma once
#include "nurbscontact/cpu_mesh.hpp"

namespace nc::cpu {
class SolidBoundary {
public:
    explicit SolidBoundary(const MeshData& mesh);
    ~SolidBoundary();
    bool contains(Vec local) const;
    std::optional<Vec> interior_projection(Vec local) const;
    std::size_t component(std::size_t face) const;
    std::array<std::size_t,3> neighbors(std::size_t face) const;
    std::span<const std::size_t> representatives() const;
    const std::array<std::size_t,6>& extreme_vertices(std::size_t component) const;
private:
    struct Data;
    std::unique_ptr<Data> data_;
};
}
