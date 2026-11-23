#include "nurbscontact/cpu_mesh.hpp"

#include <iostream>
#include <set>

using namespace nc::cpu;
using Key = std::array<std::size_t,4>;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        Triangle a{{{0,0,0},{1,0,0},{0,1,0}}};
        Triangle cross{{{.25,.25,-1},{.25,.25,1},{.75,.25,0}}};
        require(intersect_triangles(a,cross).size()==2,"transverse crossing");
        require(intersect_triangles(a,a).size()==3,"coplanar triangle");
        Triangle up{{{-2,-1,0},{2,-1,0},{0,2,0}}};
        Triangle down{{{-2,1,0},{0,-2,0},{2,1,0}}};
        require(intersect_triangles(up,down).size()==6,"coplanar hexagonal overlap");
        Triangle point{{{1,0,0},{2,0,0},{1,1,0}}};
        require(intersect_triangles(a,point).size()==1,"vertex touch");
        Triangle coplanar{{{1,0,0},{0,0,0},{0,-1,0}}};
        require(intersect_triangles(a,coplanar).size()==2,"shared edge");
        auto separate = a;
        for (auto& v : separate) v[2]=1e-15;
        require(intersect_triangles(a,separate).empty(),"parallel separated triangles");
        auto data = std::make_shared<MeshData>();
        data->vertices = {{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
        data->faces = {{0,2,1},{0,1,3},{0,3,2},{1,2,3}};
        std::vector<std::shared_ptr<const MeshData>> inputs(3,data);
        for (const auto m : {MeshMethod::aabb,MeshMethod::obb,MeshMethod::sap}) {
            MeshScene scene(inputs,m);
            std::vector<RigidPlacement> poses(3);
            poses[2].origin = {3,0,0};
            for (const double offset : {0., .25, 4., 1., 3.125, -.125}) {
                poses[1].origin = {offset,0,0};
                const double angle=offset/3;
                poses[1].basis = {{{1,0,0},{0,std::cos(angle),std::sin(angle)},{0,-std::sin(angle),std::cos(angle)}}};
                std::set<Key> expected, actual;
                for (std::size_t i=0;i<3;++i) for (std::size_t j=i+1;j<3;++j)
                    for (std::size_t f=0;f<4;++f) for (std::size_t g=0;g<4;++g) {
                        Triangle ta,tb;
                        for (std::size_t k=0;k<3;++k) {
                            ta[k]=poses[i].point(data->vertices[data->faces[f][k]]);
                            tb[k]=poses[j].point(data->vertices[data->faces[g][k]]);
                        }
                        if (!intersect_triangles(ta,tb).empty()) expected.insert({i,j,f,g});
                    }
                auto result=scene.intersect(poses);
                for (const auto& h: result.intersections) actual.insert({h.body_a,h.body_b,h.triangle_a,h.triangle_b});
                require(actual==expected && result.intersections.size()==actual.size(),"accelerated vs exhaustive triangle pairs");
            }
            auto tiny = std::make_shared<MeshData>(*data);
            for (auto& p : tiny->vertices) for (auto& x : p) x = .1 + .1 * x;
            std::array<std::shared_ptr<const MeshData>,2> nested{data,tiny};
            MeshScene nested_scene(nested,m);
            const auto contained = nested_scene.intersect(std::array<RigidPlacement,2>{});
            require(contained.intersections.empty() && contained.enclosed.size()==1 &&
                    contained.enclosed[0].body==1 && contained.enclosed[0].enclosing_body==0,
                    "solid containment without a boundary crossing");
            auto two = std::make_shared<MeshData>(*tiny);
            for (auto p : data->vertices) {
                p[0] += .4;
                two->vertices.push_back(p);
            }
            for (auto f : data->faces) {
                for (auto& index : f) index += 4;
                two->faces.push_back(f);
            }
            nested[1]=two;
            MeshScene mixed_scene(nested,m);
            const auto mixed=mixed_scene.intersect(std::array<RigidPlacement,2>{});
            require(!mixed.intersections.empty() && mixed.enclosed.size()==1 && mixed.enclosed[0].body==1,
                    "enclosure and crossing in different components of one body");
        }
        std::cout << "CPU triangle intersections passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
