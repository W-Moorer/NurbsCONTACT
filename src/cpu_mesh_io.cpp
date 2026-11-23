#include "nurbscontact/cpu_mesh.hpp"

#include <fstream>
#include <locale>
#include <sstream>
#include <unordered_map>

namespace nc::cpu {

MeshMethod mesh_method(std::string_view name) {
    if (name == "mesh-aabb") return MeshMethod::aabb;
    if (name == "mesh-obb") return MeshMethod::obb;
    if (name == "mesh-sap") return MeshMethod::sap;
    throw std::invalid_argument("unknown CPU mesh method");
}

MeshData read_gmsh(const std::filesystem::path& file) {
    std::ifstream input(file);
    input.imbue(std::locale::classic());
    if (!input) throw std::runtime_error("cannot open Gmsh file: " + file.string());
    const auto expect = [&](std::string_view wanted) {
        std::string token;
        if (!(input >> token) || token != wanted) throw std::runtime_error("expected " + std::string(wanted));
    };
    expect("$MeshFormat");
    expect("2.2");
    expect("0");
    expect("8");
    expect("$EndMeshFormat");
    MeshData mesh;
    std::unordered_map<std::size_t, std::size_t> indices;
    std::string section, line;
    while (input >> section) {
        if (section == "$Nodes") {
            std::size_t count{};
            if (!(input >> count) || !mesh.vertices.empty()) throw std::runtime_error("invalid node section");
            for (std::size_t i = 0; i < count; ++i) {
                std::size_t tag{};
                Vec point{};
                if (!(input >> tag >> point[0] >> point[1] >> point[2]) ||
                    !std::ranges::all_of(point, [](double x) { return std::isfinite(x); }) ||
                    !indices.emplace(tag, i).second) throw std::runtime_error("invalid mesh node");
                mesh.vertices.push_back(point);
            }
            expect("$EndNodes");
        } else if (section == "$Elements") {
            std::size_t count{};
            if (!(input >> count) || !mesh.faces.empty()) throw std::runtime_error("invalid element section");
            std::getline(input, line);
            for (std::size_t i = 0; i < count; ++i) {
                if (!std::getline(input, line)) throw std::runtime_error("missing element");
                std::istringstream row(line);
                std::size_t id{};
                int type{}, tags{};
                if (!(row >> id >> type >> tags) || tags < 0) throw std::runtime_error("invalid mesh element");
                if (type == 15 || type == 1) continue;
                if (type != 2) throw std::runtime_error("expected a first-order triangle surface mesh");
                int face = 0;
                for (int j = 0; j < tags; ++j) {
                    int tag{};
                    if (!(row >> tag)) throw std::runtime_error("incomplete mesh tags");
                    if (j == 1) face = tag;
                }
                std::array<std::size_t, 3> tri{};
                for (auto& v : tri) {
                    std::size_t tag{};
                    if (!(row >> tag) || !indices.contains(tag)) throw std::runtime_error("unknown triangle vertex");
                    v = indices.at(tag);
                }
                mesh.faces.push_back(tri);
                mesh.cad_faces.push_back(face);
            }
            expect("$EndElements");
        } else {
            if (!section.starts_with('$') || section.starts_with("$End")) throw std::runtime_error("bad Gmsh section");
            const auto end = "$End" + section.substr(1);
            while (input >> section && section != end) {}
            if (!input) throw std::runtime_error("unterminated Gmsh section");
        }
    }
    if (mesh.faces.empty()) throw std::runtime_error("Gmsh file has no triangles");
    return mesh;
}

} // namespace nc::cpu
