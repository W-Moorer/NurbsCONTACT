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
            // Splitting an original face adds mesh edges on exactly the same plane.
            auto split=std::make_shared<MeshData>(*data);
            split->vertices.push_back({.25,.25,0});
            split->faces[0]={0,2,4};
            split->faces.push_back({2,1,4});
            split->faces.push_back({1,0,4});
            std::array<RigidPlacement,2> contact{};
            contact[1].origin={.125,.125,-.25};
            for (auto surface : {data,split}) {
                std::array<std::shared_ptr<const MeshData>,2> pair{surface,data};
                MeshScene connected(pair,m);
                const auto patches=connected.intersect(contact,true);
                require(patches.contours.size()==1 && patches.contours[0].triangles.size()==patches.intersections.size(),
                        "artificial face split divided a connected intersection");
            }
            auto split_loaded=std::make_shared<MeshData>(*data);
            split_loaded->vertices.push_back({.25,.25,.5});
            split_loaded->faces[3]={1,2,4};
            split_loaded->faces.push_back({2,3,4});
            split_loaded->faces.push_back({3,1,4});
            for (auto surface : {data,split_loaded}) {
                std::array<std::shared_ptr<const MeshData>,2> pair{surface,data};
                MeshScene scene(pair,m);
                std::array<RigidPlacement,2> pose{};
                pose[1].origin={.125,.125,.25};
                // The exposed portion of A is on x+y+z=1 and projects to
                // a triangle of area (1-.125-.125-.25)^2/2 on each axis plane.
                for (const auto sign : {1.,-1.}) {
                    const auto geometry=scene.intersect(pose,true);
                    require(geometry.contours.size()==1 && geometry.contours[0].vector_area_a_m2.has_value(),
                            "missing closed transverse loop");
                    for (double component : *geometry.contours[0].vector_area_a_m2)
                        require(std::abs(component-sign*.125)<1e-12,"incorrect oriented vector area");
                    std::swap(pose[0],pose[1]);
                }
            }
            auto islands=std::make_shared<MeshData>(*data);
            for (auto p : data->vertices) {
                p[0]+=3.;
                islands->vertices.push_back(p);
            }
            for (auto f : data->faces) {
                for (auto& index : f) index+=4;
                islands->faces.push_back(f);
            }
            std::array<std::shared_ptr<const MeshData>,2> island_pair{islands,islands};
            MeshScene island_scene(island_pair,m);
            require(island_scene.intersect(contact,true).contours.size()==2,"independent contact contours were merged");
#ifdef NC_WITH_MESH_CCD
            std::array<std::shared_ptr<const MeshData>,2> moving{data,data};
            MeshScene swept(moving,m);
            std::array<RigidPlacement,2> start{},end{};
            start[1].origin={-2,0,0};
            end[1].origin={2,0,0};
            require(swept.intersect(start).intersections.empty() && swept.intersect(end).intersections.empty(),
                    "swept fixture endpoints must be clear");
            auto crossing=swept.sweep(start,end,1e-7);
            require(crossing.first && std::abs(crossing.first->fraction-.25)<1e-12 &&
                    crossing.translation_tests>0 && crossing.first->geometric_tolerance_m<1e-12,"translation TOI");
            end[1].origin={-3,0,0};
            require(!swept.sweep(start,end,1e-7).first,"separating motion");
            end=start;
            for (auto& pose : end) pose.origin[1]+=3.;
            require(!swept.sweep(start,end,1e-7).first,"common translation must preserve separation");
            start[1].origin={.2,.2,.2};
            end=start;
            end[1].origin={3.,0.,0.};
            const auto initial=swept.sweep(start,end,1e-7);
            require(initial.first && initial.first->fraction==0.,"initial boundary intersection");
            auto off_axis=std::make_shared<MeshData>(*data);
            for (auto& p : off_axis->vertices) {
                for (auto& x : p) x *= .05;
                p[0] += 1.;
            }
            moving={off_axis,off_axis};
            MeshScene rotating(moving,m);
            start={}; end={};
            const auto spin=[](double angle) {
                RigidPlacement pose;
                const auto c=std::cos(angle), s=std::sin(angle);
                pose.basis={{{c,s,0},{-s,c,0},{0,0,1}}};
                return pose;
            };
            start[1]=spin(-1.);
            end[1]=spin(1.);
            require(rotating.intersect(start).intersections.empty() && rotating.intersect(end).intersections.empty(),
                    "arc fixture endpoints must be clear");
            auto arc=rotating.sweep(start,end,1e-5);
            // The rotating vertex (1,.05,0) first meets the fixed edge y=z=0.
            const double expected_fraction=(1.-std::atan(.05))/2.;
            require(arc.first && std::abs(arc.first->fraction-expected_fraction)<2e-5 &&
                    arc.first->geometric_tolerance_m<=1e-5,"interior rotation event");
            start[1].origin[2]=end[1].origin[2]=.052;
            require(!rotating.sweep(start,end,1e-5).first,"rotation past a separated parallel face");
#endif
        }
        std::cout << "CPU triangle intersections passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
