#include "cpu_mesh_solid.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Surface_mesh.h>
#include <numeric>

namespace nc::cpu {
using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Surface = CGAL::Surface_mesh<K::Point_3>;
using Side = CGAL::Side_of_triangle_mesh<Surface,K>;

struct SolidBoundary::Data {
    Surface surface;
    std::unique_ptr<Side> side;
    std::vector<std::size_t> representatives;
    std::vector<std::size_t> face_components;
    std::vector<std::array<std::size_t,6>> extremes;
};

SolidBoundary::SolidBoundary(const MeshData& mesh) : data_(std::make_unique<Data>()) {
    std::vector<std::size_t> parent(mesh.vertices.size());
    std::iota(parent.begin(),parent.end(),0);
    const auto root = [&](std::size_t i) {
        while (i != parent[i]) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    for (Vec p : mesh.vertices) data_->surface.add_vertex({p[0],p[1],p[2]});
    for (const auto& face : mesh.faces) {
        auto vertex = [](std::size_t i) { return Surface::Vertex_index(static_cast<unsigned int>(i)); };
        if (data_->surface.add_face(vertex(face[0]),vertex(face[1]),vertex(face[2])) == Surface::null_face())
            throw std::invalid_argument("inconsistent or non-manifold contact mesh");
        parent[root(face[1])] = root(face[0]);
        parent[root(face[2])] = root(face[0]);
    }
    if (!CGAL::is_closed(data_->surface)) throw std::invalid_argument("contact mesh has a boundary edge");
    std::vector<std::size_t> components(parent.size(),mesh.faces.size());
    for (const auto& face : mesh.faces) {
        const auto component = root(face[0]);
        if (components[component] == mesh.faces.size()) {
            components[component] = data_->representatives.size();
            data_->representatives.push_back(face[0]);
            data_->extremes.push_back({});
            data_->extremes.back().fill(face[0]);
        }
        data_->face_components.push_back(components[component]);
    }
    for (std::size_t i=0;i<parent.size();++i) {
        const auto component=components[root(i)];
        if (component==mesh.faces.size()) continue;
        auto& ends=data_->extremes[component];
        for (std::size_t axis=0;axis<3;++axis) {
            if (mesh.vertices[i][axis]<mesh.vertices[ends[2*axis]][axis]) ends[2*axis]=i;
            if (mesh.vertices[i][axis]>mesh.vertices[ends[2*axis+1]][axis]) ends[2*axis+1]=i;
        }
    }
    data_->side = std::make_unique<Side>(data_->surface);
}
SolidBoundary::~SolidBoundary() = default;
bool SolidBoundary::contains(Vec p) const {
    return (*data_->side)(K::Point_3{p[0],p[1],p[2]}) == CGAL::ON_BOUNDED_SIDE;
}
std::span<const std::size_t> SolidBoundary::representatives() const { return data_->representatives; }
const std::array<std::size_t,6>& SolidBoundary::extreme_vertices(std::size_t component) const {
    return data_->extremes.at(component);
}
std::size_t SolidBoundary::component(std::size_t face) const { return data_->face_components.at(face); }
}
