#include "nurbscontact/cpu_mesh.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <set>

using namespace nc::cpu;
using Key = std::array<std::size_t,4>;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void append_box(MeshData& mesh,double left,double right) {
    const std::size_t start=mesh.vertices.size();
    for (const double z : {0.,1.})
        for (const double y : {0.,1.})
            for (const double x : {left,right}) mesh.vertices.push_back({x,y,z});
    const std::array<std::array<std::size_t,3>,12> faces{{
        {0,2,3},{0,3,1},{4,5,7},{4,7,6},{0,1,5},{0,5,4},
        {2,6,7},{2,7,3},{0,4,6},{0,6,2},{1,3,7},{1,7,5}}};
    for (auto f : faces) {
        for (auto& id : f) id+=start;
        mesh.faces.push_back(f);
    }
}

void check_penalty_regions() {
    using namespace nc::dynamics;
    auto box=std::make_shared<MeshData>();
    append_box(*box,0.,1.);
    auto upper=std::make_shared<MeshData>(*box);
    upper->vertices.push_back({.25,.5,0.});
    upper->faces[0]={0,2,8};
    upper->faces.push_back({2,3,8});
    upper->faces.push_back({3,0,8});
    ContactPose resting,pressed;
    pressed.origin[2]=.75;
    ContactWork work;
    for (auto method : {MeshMethod::aabb,MeshMethod::obb,MeshMethod::sap}) {
        auto query=mesh_region_query(box,upper,method);
        const auto one=query->regions(resting,pressed,work);
        require(one.size()==1 && std::abs(one[0].depth_m-.25)<1e-12 &&
                length(sub(one[0].point,{.5,.5,.875}))<1e-12 &&
                length(sub(one[0].normal,{0,0,1}))<1e-12,"split box face generated incorrect overlap region");
        const auto inverse=mesh_region_query(upper,box,method)->regions(pressed,resting,work);
        require(inverse.size()==1 && length(add(inverse[0].normal,one[0].normal))<1e-12 &&
                length(sub(inverse[0].point,one[0].point))<1e-12,"asymmetric mesh penalty region");
        ContactPose touch;
        touch.origin[2]=1.;
        require(query->regions(resting,touch,work).empty(),"touch without volume must carry no elastic load");
#ifdef NC_WITH_MESH_CCD
        ContactPose left,right;
        left.origin[2]=-2.; right.origin[2]=2.;
        const auto tunnel=query->transition(resting,left,resting,right,1e-7);
        require(tunnel.fraction && std::abs(*tunnel.fraction-.25)<1e-10 &&
                !tunnel.start_in_contact && !tunnel.end_in_contact,"missed a crossing between disjoint endpoints");
        const auto release=query->transition(resting,pressed,resting,right,1e-7);
        require(release.fraction && std::abs(*release.fraction-.2)<1e-10 &&
                release.start_in_contact && !release.end_in_contact,"incorrect release on a reversed sweep");
#endif
    }
#ifdef NC_WITH_MESH_CCD
    const nc::Vec center{-2.,.5,.5};
    auto pivot_query=mesh_region_query(box,box,MeshMethod::obb,{nc::Vec{},center});
    const nc::dynamics::ContactPose target{{2.,-.5,-.5},{0,0,1},0};
    auto orbit=[&](double angle) {
        nc::dynamics::ContactPose pose{{},{0,0,1},angle};
        pose.origin=scale(pose.rotate(center),-1.);
        return pose;
    };
    const auto entry=pivot_query->transition(target,orbit(-1.),target,orbit(1.),1e-6);
    require(!entry.start_in_contact && !entry.end_in_contact && entry.fraction &&
            *entry.fraction>0 && *entry.fraction<.5,"offset pivot lost the interior contact on a rotating arc");
#endif
    auto twin=std::make_shared<MeshData>(*box),plate=std::make_shared<MeshData>();
    append_box(*twin,2.,3.);
    append_box(*plate,-.5,3.5);
    auto query=mesh_region_query(twin,plate,MeshMethod::aabb);
    const auto two=query->regions(resting,pressed,work);
    require(two.size()==2 && std::abs(two[0].depth_m-.25)<1e-12 && std::abs(two[1].depth_m-.25)<1e-12,
            "two disconnected overlap volumes must carry separate loads");
}

void check_triangle_force_geometry() {
    using namespace nc::dynamics;
    auto block=std::make_shared<MeshData>(),tip=std::make_shared<MeshData>();
    append_box(*block,0,1);
    tip->vertices={{.65,.25,0},{.60,.20,.20},{.70,.20,.20},{.65,.30,.20}};
    tip->faces={{0,2,1},{0,3,2},{0,1,3},{1,2,3}};
    ContactPose base,pose;
    ContactWork work;
    auto query=mesh_region_query(block,tip,MeshMethod::obb,{},MeshRegionMode::fcl);
    ContactPose face_touch;
    face_touch.origin[2]=1;
    auto cubes=mesh_region_query(block,block,MeshMethod::obb,{},MeshRegionMode::fcl);
    require(cubes->regions(base,face_touch,work).empty(),"coplanar faces must not load their common rim");
    pose.origin[2]=1.03;
    require(query->regions(base,pose,work).empty(),"disjoint FCL pair produced a force");
    pose.origin[2]=1;
    require(query->regions(base,pose,work).empty(),"touching FCL tip produced a force");
    pose.origin[2]=.98;
    const auto hits=query->regions(base,pose,work);
    require(hits.size()==1 && std::abs(hits[0].depth_m-.02)<1e-12 &&
            length(sub(hits[0].normal,{0,0,1}))<1e-12 && length(sub(hits[0].point,{.65,.25,.98}))<1e-12,
            "FCL tip-plane depth, normal or contact point");
    const auto finite=mesh_region_query(block,tip,MeshMethod::obb,{},MeshRegionMode::finite,.03);
    const auto bounded=finite->regions(base,pose,work);
    require(bounded.size()==1 && std::abs(bounded[0].depth_m-.02)<1e-12 &&
            length(sub(bounded[0].normal,{0,0,1}))<1e-10,"finite vertex-face penetration");
    auto separated=pose;
    separated.origin[2]=1.05;
    require(finite->regions(base,separated,work).empty(),"stale finite contact on separation");
    const auto reverse=mesh_region_query(tip,block,MeshMethod::obb,{},MeshRegionMode::finite,.03)->regions(pose,base,work);
    require(reverse.size()==1 && length(add(reverse[0].normal,bounded[0].normal))<1e-10 &&
            std::abs(reverse[0].depth_m-bounded[0].depth_m)<1e-12,"finite contact body reversal");
    auto twin=std::make_shared<MeshData>(*tip);
    for (auto v:tip->vertices) {v[0]-=.4; twin->vertices.push_back(v);}
    for (auto f:tip->faces) {
        for (auto& i:f) i+=tip->vertices.size();
        twin->faces.push_back(f);
    }
    for (auto method:{MeshMethod::aabb,MeshMethod::obb}) {
        auto caps=mesh_region_query(block,twin,method,{},MeshRegionMode::finite,.03);
        auto regions=caps->regions(base,pose,work);
        require(regions.size()==2 && std::abs(regions[0].depth_m-.02)<1e-12 &&
                std::abs(regions[1].depth_m-.02)<1e-12 &&
                std::abs(length(sub(regions[0].point,regions[1].point))-.4)<1e-12,"disconnected finite support caps");
        auto enclosed=pose;
        enclosed.origin[2]=.4;
        bool rejected=false;
        try {caps->regions(base,enclosed,work);} catch (const UnresolvedContact&) {rejected=true;}
        require(rejected,"fully enclosed components must not be loaded at an arbitrary planar face");
    }
    const auto inverse=mesh_region_query(tip,block,MeshMethod::obb,{},MeshRegionMode::fcl)->regions(pose,base,work);
    require(inverse.size()==1 && std::abs(hits[0].depth_m-inverse[0].depth_m)<1e-12 &&
            length(add(hits[0].normal,inverse[0].normal))<1e-12 && work.boolean_queries==0,
            "FCL body-order symmetry or unintended solid intersection");
}

void check_crossed_edges() {
    using namespace nc::dynamics;
    const auto ridge=[](std::vector<nc::Vec> vertices) {
        auto mesh=std::make_shared<MeshData>();
        mesh->vertices=std::move(vertices);
        for (std::size_t i=0;i<4;++i) {
            std::array<std::size_t,3> f{(i+1)%4,(i+2)%4,(i+3)%4};
            const auto& v=mesh->vertices;
            if (dot3(cross3(sub(v[f[1]],v[f[0]]),sub(v[f[2]],v[f[0]])),sub(v[i],v[f[0]]))>0)
                std::swap(f[0],f[1]);
            mesh->faces.push_back(f);
        }
        return mesh;
    };
    auto a=ridge({{-1,0,0},{1,0,0},{0,-.4,-1},{0,.4,-1}});
    auto b=ridge({{0,-1,0},{0,1,0},{-.4,0,1},{.4,0,1}});
    for (auto method:{MeshMethod::aabb,MeshMethod::obb}) {
        auto query=mesh_region_query(a,b,method,{},MeshRegionMode::finite,.03);
        ContactPose base,pose;
        ContactWork work;
        pose.origin[2]=-.02;
        const auto contacts=query->regions(base,pose,work);
        require(contacts.size()==1 && std::abs(contacts[0].depth_m-.02)<1e-12 &&
                length(sub(contacts[0].point,{0,0,-.01}))<1e-12 &&
                length(sub(contacts[0].normal,{0,0,1}))<1e-12,"interior edge-edge contact witness");
        for (double angle : {.13,.72,1.71,3.25,5.37}) {
            ContactPose frame{{.7,-.4,.2},{.3/std::sqrt(.5),-.4/std::sqrt(.5),.5/std::sqrt(.5)},angle};
            auto moved=frame;
            moved.origin=frame.point({0,0,-.02});
            const auto transformed=query->regions(frame,moved,work);
            require(transformed.size()==1 && std::abs(transformed[0].depth_m-.02)<1e-10 &&
                    length(sub(transformed[0].point,frame.point({0,0,-.01})))<1e-10 &&
                    length(sub(transformed[0].normal,frame.rotate({0,0,1})))<1e-10,
                    "rigid frame changed finite edge contact");
        }
        pose.origin[2]=.02;
        require(query->regions(base,pose,work).empty(),"separated crossed edges produced a load");
    }
    const auto compound=[](const MeshData& one) {
        auto mesh=std::make_shared<MeshData>();
        for (double offset:{3.,-3.,0.}) {
            auto first=mesh->vertices.size();
            for (auto p:one.vertices) {p[0]+=offset; mesh->vertices.push_back(p);}
            for (auto f:one.faces) {
                for (auto& i:f) i+=first;
                mesh->faces.push_back(f);
            }
        }
        return mesh;
    };
    const auto left=compound(*a),right=compound(*b);
    ContactPose base,pose;
    pose.origin[2]=-.02;
    ContactWork work;
    const auto contacts=[&](MeshMethod method) {
        return mesh_region_query(left,right,method,{},MeshRegionMode::finite,.03)->regions(base,pose,work);
    };
    const auto aa=contacts(MeshMethod::aabb),bb=contacts(MeshMethod::obb);
    require(aa.size()==3 && bb.size()==3,"missing disconnected edge contacts");
    for (auto method:{MeshMethod::aabb,MeshMethod::obb}) {
        const auto rays=mesh_region_query(left,right,method,{},MeshRegionMode::contour_ray,.03)->regions(base,pose,work);
        require(rays.size()==3,"contour-ray combined independent supports");
        for (const auto& region:rays)
            require(std::abs(region.depth_m-.02)<1e-10 && std::abs(region.point[2]+.01)<1e-10 &&
                    length(sub(region.normal,{0,0,1}))<1e-9,"crossing ridges changed the analytic wrench");
    }
    for (std::size_t i=0;i<aa.size();++i)
        require(aa[i].point==bb[i].point && aa[i].normal==bb[i].normal && aa[i].depth_m==bb[i].depth_m,
                "BVH traversal changed contact accumulation order");
    // Revisit rejected poses and move different components into contact.
    for (auto method:{MeshMethod::aabb,MeshMethod::obb}) {
        auto warm=mesh_region_query(left,right,method,{},MeshRegionMode::finite,.03);
        ContactWork cached;
        for (double x:{0.,.001,.06,.061,2.,6.,0.,.001}) {
            pose.origin[0]=x;
            const auto current=warm->regions(base,pose,cached);
            const auto cold=contacts(method);
            require(current.size()==cold.size(),"stale contact candidate list");
            for (std::size_t i=0;i<cold.size();++i)
                require(current[i].point==cold[i].point && current[i].normal==cold[i].normal &&
                        current[i].depth_m==cold[i].depth_m,"stale contact geometry after pose change");
        }
        require(cached.mesh_patch_reuses>0 && cached.mesh_pair_bvh_builds>1,
                "test must exercise finite contact reuse and invalidation");
    }
}

void check_local_regions() {
    using namespace nc::dynamics;
    auto mesh=std::make_shared<MeshData>();
    append_box(*mesh,0,1);
    for (int level=0;level<3;++level) {
        const auto previous=std::move(mesh->faces);
        mesh->faces.clear();
        std::map<std::pair<std::size_t,std::size_t>,std::size_t> midpoints;
        const auto midpoint=[&](std::size_t a,std::size_t b) {
            const auto [it,new_edge]=midpoints.emplace(std::minmax(a,b),mesh->vertices.size());
            if (new_edge) mesh->vertices.push_back(scale(add(mesh->vertices[a],mesh->vertices[b]),.5));
            return it->second;
        };
        for (const auto [a,b,c]:previous) {
            const auto ab=midpoint(a,b),bc=midpoint(b,c),ca=midpoint(c,a);
            mesh->faces.insert(mesh->faces.end(),{{a,ab,ca},{ab,b,bc},{ca,bc,c},{ab,bc,ca}});
        }
    }
    auto local=mesh_region_query(mesh,mesh,MeshMethod::obb);
    auto full=mesh_region_query(mesh,mesh,MeshMethod::obb,{},false);
    {
        std::array<ContactBodyState,3> states{};
        states[1].pose.origin={.997,.01,.02};
        states[2].pose.origin={.995,.02,.01};
        std::array<ContactPair,2> pairs{};
        for (unsigned i=0;i<2;++i) {
            pairs[i].b=i+1;
            pairs[i].material={1000.,1.,0.,0.,.01,.05};
            pairs[i].regions=mesh_region_query(mesh,mesh,MeshMethod::obb,{},MeshRegionMode::finite,.05);
        }
        ContactWork independent,shared;
        std::vector<ContactSample> batch,serial;
        require(add_contact_forces(nullptr,states,pairs,{},independent,&batch),"independent mesh queries failed");
        pairs[1].regions=pairs[0].regions;
        require(add_contact_forces(nullptr,states,pairs,{},shared,&serial),"shared query failed");
        require(shared.mesh_parallel_batches==0 && serial.size()==2 && batch.size()==2,
                "queries sharing a geometry cache must execute serially");
        for (int i=0;i<2;++i)
            require(serial[i].normal_force>0 && serial[i].normal_force==batch[i].normal_force &&
                    serial[i].gap==batch[i].gap,"batch assembly changed the mesh load");
    }
    std::array<std::shared_ptr<const MeshData>,2> mesh_pair{mesh,mesh};
    {
        MeshScene sliding(mesh_pair,MeshMethod::obb);
        std::array<RigidPlacement,2> poses{};
        bool advanced=false;
        for (int i=0;i<=100;++i) {
            poses[1].origin={.997,-.75+.015*i,.037};
            const auto hit=sliding.overlaps(poses);
            require(hit.overlap,"lost overlap during a tangential slide");
            advanced|=hit.adjacent_triangle_pair;
        }
        require(advanced,"did not follow an intersecting pair onto neighboring triangles");
        poses[1].origin[0]=1.002;
        require(!sliding.overlaps(poses).overlap,"accepted a stale witness after separation");
    }
    for (auto method:{MeshMethod::aabb,MeshMethod::obb,MeshMethod::sap}) {
        MeshScene geometry(mesh_pair,method,true);
        std::array<RigidPlacement,2> poses{};
        poses[1].origin={1.007,0,0};
        ContactWork near_work;
        const auto candidates=geometry.near_pairs(poses,.01,near_work);
        const std::set<std::array<std::size_t,2>> nearby(candidates.begin(),candidates.end());
        std::size_t witnessed{};
        for (std::size_t a=0;a<mesh->faces.size();++a) {
            const auto i=mesh->faces[a][0];
            const auto p=mesh->vertices[i];
            if (p[0]!=1.) continue;
            for (std::size_t b=0;b<mesh->faces.size();++b) {
                const auto q=poses[1].point(mesh->vertices[mesh->faces[b][0]]);
                if (length(sub(p,q))>.01) continue;
                ++witnessed;
                require(nearby.contains({a,b}),"positive-margin window discarded a close vertex pair");
            }
        }
        require(witnessed>0,"missing proximity fixture");
        for (double shift:{.002,.012,.31,.005}) {
            poses[1].origin={.997,shift,.003};
            const auto pairs=geometry.near_pairs(poses,.01,near_work);
            const std::set<std::array<std::size_t,2>> present(pairs.begin(),pairs.end());
            const auto truth=geometry.intersect(poses);
            require(!truth.intersections.empty(),"sliding boundary fixture lost contact");
            for (const auto& hit:truth.intersections)
                require(present.contains({hit.triangle_a,hit.triangle_b}),"cached surface window lost a new contact");
        }
        require(near_work.mesh_pair_bvh_reuses>0 && near_work.mesh_pair_bvh_builds>1,
                "sliding surface window must reuse and refresh its candidates");
        // Each former intersection supplies two material points exactly .009
        // apart after translation, including intersections inside a facet.
        auto skew=poses;
        const double angle=.37;
        skew[1].basis={Vec{1,0,0},Vec{0,std::cos(angle),std::sin(angle)},
                       Vec{0,-std::sin(angle),std::cos(angle)}};
        skew[1].origin={.998,.13,-.07};
        const auto crossed=geometry.intersect(skew);
        require(!crossed.intersections.empty(),"missing skew proximity witnesses");
        skew[1].origin[0]+=.009;
        const auto separated=geometry.near_pairs(skew,.0091,near_work);
        const std::set<std::array<std::size_t,2>> close(separated.begin(),separated.end());
        for (const auto& hit:crossed.intersections)
            require(close.contains({hit.triangle_a,hit.triangle_b}),"clipped window lost a close facet pair");
        for (int i=0;i<12;++i) {
            const double angle=.1*i;
            poses[1].origin={.98+.013*i,.025,.031};
            poses[1].basis={{{std::cos(angle),std::sin(angle),0},{-std::sin(angle),std::cos(angle),0},{0,0,1}}};
            const auto all=geometry.intersect(poses);
            require(geometry.overlaps(poses).overlap==
                    (!all.intersections.empty() || !all.enclosed.empty()),"contact-window pruning changed overlap");
        }
    }
    ContactWork work,reference;
    for (Vec offset:std::vector<Vec>{{.96,.031,.048},{.95,.032,.047},{-.96,.031,.048},
                                     {.021,.97,-.024},{2.,0,0},{.035,.012,.017}}) {
        ContactPose a,b;
        b.origin=offset;
        b.angle=.013;
        const auto result=local->regions(a,b,work),expected=full->regions(a,b,reference);
        require(result.size()==expected.size(),"local solid changed overlap connectivity on changing sides");
        if (!expected.empty()) require(length(sub(result[0].point,expected[0].point))<1e-10 &&
            length(sub(result[0].normal,expected[0].normal))<1e-10 &&
            std::abs(result[0].depth_m-expected[0].depth_m)<1e-10,"local solid changed contact geometry");
    }
    require(work.local_solid_uses>=4 && work.boolean_input_faces<reference.boolean_input_faces,
            "localized box queries never reduced the constructive geometry");
}

int main() {
    try {
        check_penalty_regions();
        check_triangle_force_geometry();
        check_crossed_edges();
        check_local_regions();
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
                const auto any=scene.overlaps(poses);
                require(any.overlap==(!expected.empty() || !result.enclosed.empty()),
                        "overlap decision disagrees with exhaustive surface intersections and containment");
                if (!expected.empty()) {
                    const auto repeated=scene.overlaps(poses);
                    require(repeated.overlap && repeated.reused_triangle_pair,
                            "a repeated surface overlap should reuse its verified triangle pair");
                }
                for (const auto& h: result.intersections) actual.insert({h.body_a,h.body_b,h.triangle_a,h.triangle_b});
                require(actual==expected && result.intersections.size()==actual.size(),"accelerated vs exhaustive triangle pairs");
            }
            auto tiny = std::make_shared<MeshData>(*data);
            for (auto& p : tiny->vertices) for (auto& x : p) x = .1 + .1 * x;
            std::array<std::shared_ptr<const MeshData>,2> nested{data,tiny};
            MeshScene nested_scene(nested,m);
            const auto contained = nested_scene.intersect(std::array<RigidPlacement,2>{});
            require(nested_scene.overlaps(std::array<RigidPlacement,2>{}).overlap,
                    "predicate-only query lost a completely enclosed body");
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
            {
                std::array<std::shared_ptr<const MeshData>,2> forward{data,split},backward{split,data};
                MeshScene left(forward,m),right(backward,m);
                auto reverse_pose=contact;
                std::swap(reverse_pose[0],reverse_pose[1]);
                std::set<std::pair<std::size_t,std::size_t>> left_faces,right_faces;
                for (const auto& p:left.intersect(contact).intersections) left_faces.emplace(p.triangle_a,p.triangle_b);
                for (const auto& p:right.intersect(reverse_pose).intersections) right_faces.emplace(p.triangle_b,p.triangle_a);
                require(!left_faces.empty() && left_faces==right_faces,"mixed-resolution BVH changed original face identities");
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
