#include "../src/compiled_geometry.hpp"
#ifdef NC_WITH_CUDA_COLUMNS
#include "../src/cuda_columns.hpp"
#endif
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>

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

void append_box(PartFile &file, Vec offset = {}, Vec half = {1,1,1}) {
    if (file.geometry.trims.empty()) file.geometry.trims.emplace_back();
    for (int axis = 0; axis < 3; ++axis)
        for (int sign : {-1, 1}) {
            const int u=(axis+1)%3,v=(axis+2)%3;
            Rect domain{{-half[u],half[u]},{-half[v],half[v]}};
            Vec center=offset, x{}, y{}, lo{}, hi{};
            for (int k=0;k<3;++k) {lo[k]=offset[k]-half[k];hi[k]=offset[k]+half[k];}
            center[axis] = lo[axis] = hi[axis] = offset[axis]+sign*half[axis];
            x[u] = 1;
            y[v] = sign;
            file.faces.push_back({domain, lo, hi, 1e-7, 1, true});
            file.geometry.patches.push_back(
                {Plane{point(center), point(x), point(y)}, domain, file.faces.size(), 0});
        }
}
PartFile box() {
    PartFile file;
    append_box(file);
    return file;
}
// Assign exact shared topology to a synthetic collection of planar charts.
// Used only in tests; production uses the original importer feature IDs.
void planar_topology(PartFile &file) {
    std::map<Vec,std::size_t> vertices;
    std::map<std::pair<std::size_t,std::size_t>,std::size_t> edges;
    for (auto &patch:file.geometry.patches) {
        const auto r=patch.domain;
        const std::array<Vec,4> uv{{{r.u.lo,r.v.lo,0},{r.u.hi,r.v.lo,0},
                                   {r.u.hi,r.v.hi,0},{r.u.lo,r.v.hi,0}}};
        std::array<std::size_t,4> ids;
        for (int k=0;k<4;++k) {
            const auto p=evaluate(patch.surface,{Interval{uv[k][0]},Interval{uv[k][1]}});
            Vec xyz{p[0].center(),p[1].center(),p[2].center()};
            ids[k]=vertices.try_emplace(xyz,vertices.size()+1).first->second;
        }
        TrimDomain trim; trim.loops.emplace_back();
        for (int k=0;k<4;++k) {
            int j=(k+1)%4;
            const auto key=std::minmax(ids[k],ids[j]);
            const auto id=edges.try_emplace(key,edges.size()+1).first->second;
            trim.loops.back().push_back({BezierCurve{{homogeneous(point(uv[k])),homogeneous(point(uv[j]))}},
                {FeatureKind::edge,id},{FeatureKind::vertex,ids[k]},{FeatureKind::vertex,ids[j]}});
        }
        patch.trim_id=file.geometry.trims.size(); file.geometry.trims.push_back(std::move(trim));
    }
}
#ifdef NC_WITH_CUDA_COLUMNS
void describe_part(const std::filesystem::path &path) {
    const auto solid=load_solid(path);
    std::map<std::size_t,std::vector<int>> edges;
    std::map<std::size_t,bool> vertices;
    for(int id=0;id<int(solid->faces.size());++id) {
        const auto *trim=solid->faces[id].trim;
        if(!trim) continue;
        for(const auto &loop:trim->loops) for(const auto &edge:loop) {
            if(edge.feature.kind==FeatureKind::edge) {
                auto &uses=edges[edge.feature.id];
                if(std::find(uses.begin(),uses.end(),id)==uses.end()) uses.push_back(id);
            }
            for(auto v:{edge.start,edge.end}) if(v.kind==FeatureKind::vertex) vertices[v.id]=true;
        }
    }
    std::size_t shared=0;
    for(const auto &entry:edges) shared+=entry.second.size()==2;
    size_t histogram[4][4]{},spans=0,cells=0,analytic=0,rectangular=0;
    double tail_u=0,tail_v=0,scale=0;
    for (const auto &f:solid->faces) {
        analytic+=f.analytic!=nullptr; rectangular+=f.info.rectangle;
        for (const auto &p:f.patches) if(p.cubic) {
            ++spans; cells+=p.cells.size(); int u=0,v=0;
            for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int c=0;c<3;++c){
                const double value=p.power[4*i+j][c],mag=std::abs(value);
                if(value!=0){u=std::max(u,i);v=std::max(v,j);}
                scale=std::max(scale,mag);
                if(i>1)tail_u=std::max(tail_u,mag);
                if(j>1)tail_v=std::max(tail_v,mag);
            }
            ++histogram[u][v];
        }
    }
    std::cout<<std::setprecision(17)<<"{\"part\":"<<std::quoted(path.filename().string())
             <<",\"faces\":"<<solid->faces.size()<<",\"source_edges\":"<<edges.size()
             <<",\"shared_edges\":"<<shared<<",\"source_vertices\":"<<vertices.size()<<",\"analytic_faces\":"<<analytic
             <<",\"rectangular_faces\":"<<rectangular<<",\"spans\":"<<spans<<",\"cells\":"<<cells
             <<",\"tail_u\":"<<tail_u<<",\"tail_v\":"<<tail_v<<",\"scale\":"<<scale
             <<",\"degree_histogram\":[";
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)std::cout<<(i+j?",":"")<<histogram[i][j];
    std::cout<<"]}\n"<<std::flush;
}
bool audit_part(const std::filesystem::path &path) {
    const auto solid=load_solid(path);
    auto gpu=cuda_columns(*solid);
    std::vector<device::ChartPoint> points;
    std::vector<bool> boundary;
    for(size_t id=0;id<solid->faces.size();++id) {
        const auto &f=solid->faces[id];
        auto probe=[&](Rect r){
            for(Param p:{Param{.07,.81},Param{.5,.5},Param{.91,.19}}){
                points.push_back({static_cast<int>(id),r.u.lo+p[0]*r.u.width(),r.v.lo+p[1]*r.v.width()});
                boundary.push_back(false);
            }
        };
        if(f.analytic)probe(f.info.domain);
        else for(const auto &s:f.patches)probe(s.domain);
        for(const auto &edge:f.boundary)for(double t:{0.,.5,1.}){
            const auto uv=evaluate(*edge.curve,t);
            points.push_back({static_cast<int>(id),uv[0].center(),uv[1].center()});
            boundary.push_back(true);
        }
    }
    auto samples=gpu->sample_charts(points);
    size_t counts[2]{},mixed[2]{},mismatch[2]{},missing=0;
    double position=0,angle=0,fraction=0;
    auto vector=[](device::V v)->Vec{return {v.x[0],v.x[1],v.x[2]};};
    for(size_t i=0;i<points.size();++i){
        const auto &p=points[i];const auto &s=samples[i];const auto &f=solid->faces[p.face];
        const int k=boundary[i];++counts[k];
        const Param uv{p.u,p.v};
        if(s.material<0)++mixed[k];else mismatch[k]+=bool(s.material)!=f.inside(uv);
        if(!f.analytic && !f.patch(uv)){missing+=s.defined;continue;}
        if(!s.defined){++missing;continue;}
        auto ref=f.at(uv);double delta=length(sub(vector(s.point),ref.p));
        if(!std::isfinite(delta))throw std::runtime_error("nonfinite chart sample");
        position=std::max(position,delta);
        fraction=std::max(fraction,delta/f.info.tolerance);
        auto n=scale(cross3(ref.u,ref.v),f.info.orientation);auto m=vector(s.normal);
        if(length(n)>0 && length(m)>0)angle=std::max(angle,std::atan2(length(cross3(n,m)),dot3(n,m)));
        else if((length(n)>0)!=(length(m)>0))++missing;
    }
    std::cout<<std::setprecision(17)<<"{\"part\":"<<std::quoted(path.filename().string())
             <<",\"faces\":"<<solid->faces.size()<<",\"samples\":"<<counts[0]<<",\"trim_samples\":"<<counts[1]
             <<",\"max_position_mm\":"<<position<<",\"max_face_tolerance_fraction\":"<<fraction
             <<",\"max_normal_rad\":"<<angle<<",\"chart_failures\":"<<missing
             <<",\"material_mismatches\":"<<mismatch[0]<<",\"trim_mismatches\":"<<mismatch[1]
             <<",\"material_unresolved\":"<<mixed[0]<<",\"trim_unresolved\":"<<mixed[1]<<"}\n";
    return !missing && !mismatch[0] && !mismatch[1] && !mixed[0] && !mixed[1] && fraction<=1;
}
void revolved_rays() {
    const double pi=std::acos(-1.);
    for(int shape=0;shape<3;++shape){
        const double major=shape==1?0.:2.,minor=shape==0?0.:.5;
        Rect r{{-pi,pi},shape==0?Interval{-1,1}:shape==1?Interval{-pi/2,pi/2}:Interval{-pi,pi}};
        PartFile part;
        part.geometry.trims.emplace_back();
        part.faces.push_back({r,{-3,-3,-1},{3,3,1},1e-7,1,true});
        Revolved chart{point({}),point({1,0,0}),point({0,1,0}),point({0,0,1}),
                       major,shape==0?.2:0.,shape==0?1.:0.,minor};
        part.geometry.patches.push_back({chart,r,1,0});
        Solid solid(std::move(part));
        auto gpu=cuda_columns(solid,false,3.);
        const double x=shape==0?1.:major;
        std::vector<device::Hit> hits;
        gpu->trace({{{{x,0,0}},{{1,0,0}}}},3.,hits);
        const auto ref=solid.first({x,0,0},{1,0,0},3.);
        require(ref && hits.size()==1 && hits[0].face==ref->face &&
                std::abs(hits[0].t-(shape==0?1.:.5))<1e-10 && std::abs(hits[0].t-ref->t)<1e-10,
                "revolved chart ray differs from analytic exit");
    }
}
void transverse_rays() {
    // Long extrusion of z=(u-1/2)^2. A ray parallel to the extrusion must
    // resolve a miss without spending its budget slicing the long direction.
    const Rect domain{{0, 1}, {0, 1}};
    PartFile file;
    file.geometry.trims.emplace_back();
    file.faces.push_back({domain, {0, 0, -.09}, {1, 1000, .25}, 1e-7, 1, true});
    BezierSurface surface{4, 4, {}, domain};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            surface.poles.push_back(homogeneous(point(
                {i / 3., 1000. * j / 3., .25 - i / 3. + i * (i - 1) / 6.})));
    file.geometry.patches.push_back({surface, domain, 1, 0});
    auto gpu = cuda_columns(Solid(std::move(file)), false, 1.);
    const double dz = 1e-7;
    std::vector<device::Ray> rays{
        {{{.5, -1, -1e-6}}, {{0, 1, 0}}},
        {{{.25, -1, .0625 + 1e-5}}, {{0, 1, 0}}},
        {{{.25, -1, .0625 - 1e-5}}, {{0, std::sqrt(1 - dz * dz), dz}}}
    };
    std::vector<device::Hit> hits;
    gpu->trace(rays, 1000., hits);
    require(hits.size() == 3 && hits[0].face == -1 && hits[1].face == -1,
            "long transverse extrusion miss was not resolved");
    require(hits[2].face == 0 && std::abs(hits[2].t - 100.) < 1e-5,
            "near-parallel extrusion crossing differs from analytic depth");
    std::vector<device::Hit> fast;
    gpu->trace(rays, 1000., fast, true);
    for (std::size_t i=0;i<fast.size();++i) {
        if (fast[i].face==-3) continue;
        require(fast[i].face==hits[i].face,"curvature enclosure changed a grazing-ray classification");
        if (fast[i].face>=0) require(fast[i].t==hits[i].t,"curvature enclosure changed a grazing root");
    }
}
void resident_boxes() {
    const auto geometry=cuda_columns(Solid(box()),true,.4);
    device::ResidentScene scene;
    scene.geometry={geometry,geometry};
    scene.coordinates={{1,0,0,0,0,0,0,0,0}};
    const device::V zero{},x{{1,0,0}},z{{0,0,1}};
    scene.bodies={{zero,z,zero,0,-1,-1,zero},{{{.0019,0,0}},z,zero,0,-1,0,x}};
    scene.pairs={{0,1,1e4,1,0,.0002,true}};
    constexpr double dt=1e-4;
    constexpr int steps=220;
    bool warm_guard=false;
    try {
        device::advance_resident(scene,dt,steps,device::RegionReduction::ordered,
                                 device::ResidentExecution::coherent,false,121);
    } catch(const std::invalid_argument &error) {
        warm_guard=std::string_view(error.what()).find("diagnostic-only")!=std::string_view::npos;
    }
    require(warm_guard,"unvalidated warm projection escaped the diagnostic-only guard");
    auto result=device::advance_resident(scene,dt,steps);
    constexpr std::size_t width=8;
    require(result.frames.size()==(steps+1)*width,"missing resident states");
    require(std::abs(result.frames[6]-1.)<1e-10,"initial box penalty force");
    int switch_step=0;
    for(int i=0;i<=steps;++i) {
        const double *r=result.frames.data()+width*i;
        const double t=i*dt,release=.5*std::acos(-1.)/100.;
        double expected=t<=release?1e-4*(1-std::cos(100*t)):1e-4+.01*(t-release);
        require(std::isfinite(r[1]) && std::abs(r[1]-expected)<1e-8,"resident box displacement");
        require(std::abs(r[6]-1e4*std::max(0.,1e-4-r[1]))<1e-8,"resident box force");
        if(i && !r[7] && !switch_step)switch_step=i;
    }
    require(switch_step==158,"resident box release step");
    const auto cached=device::advance_resident(scene,dt,steps,device::RegionReduction::audit,
                                               device::ResidentExecution::cached);
    require(std::equal(cached.frames.begin(),cached.frames.end(),result.frames.begin(),result.frames.end(),
                       [](double a,double b){return a==b || (std::isnan(a) && std::isnan(b));}),
            "cached cooperative execution changed analytic box response");
    require(cached.separated_groups>0,"separated chart certificate was not exercised after release");
    const auto batch=device::advance_resident(scene,dt,steps,device::RegionReduction::audit,
                                              device::ResidentExecution::batched);
    require(std::equal(batch.frames.begin(),batch.frames.end(),result.frames.begin(),result.frames.end(),
                       [](double a,double b){return a==b || (std::isnan(a) && std::isnan(b));}),
            "batched execution changed analytic box response");
    require(batch.batched_rays>0 && batch.ray_batches>0,"batched ray queue was not exercised");
    const auto wave=device::advance_resident(scene,dt,steps,device::RegionReduction::audit,
                                             device::ResidentExecution::wavefront);
    require(std::equal(wave.frames.begin(),wave.frames.end(),result.frames.begin(),result.frames.end(),
                       [](double a,double b){return a==b || (std::isnan(a) && std::isnan(b));}),
            "deferred query execution changed analytic box response");
    require(wave.deferred_rays>0 && wave.precomputed_rays+wave.deferred_rays==wave.batched_rays,
            "deferred ray coverage was not exercised");
    require(wave.audited_groups>0,"queued search was not checked against the original search");

    const auto coherent=device::advance_resident(scene,dt,steps,device::RegionReduction::audit,
                                                 device::ResidentExecution::coherent);
    require(std::equal(coherent.frames.begin(),coherent.frames.end(),result.frames.begin(),result.frames.end(),
                       [](double a,double b){return a==b || (std::isnan(a) && std::isnan(b));}),
            "coherent execution changed analytic box response");
    require(coherent.tested_rule_blocks>0,"coherent caches were not exercised");
    auto uncached_scene=scene;
    const auto uncached_geometry=cuda_columns(Solid(box()),false,.4);
    uncached_scene.geometry={uncached_geometry,uncached_geometry};
    const auto uncached=device::advance_resident(uncached_scene,dt,2,device::RegionReduction::audit,
                                                device::ResidentExecution::coherent);
    require(std::equal(uncached.frames.begin(),uncached.frames.end(),result.frames.begin(),
                       [](double a,double b){return a==b || (std::isnan(a) && std::isnan(b));}),
            "rule enclosure setup requires an optional low-order quadrature cache");
    // One or three disjoint analytic box contacts exercise the parallel merge
    // and the global sorting network used for larger lists.
    for (int pads : {1, 3, 5}) {
        auto sorted_scene = scene;
        PartFile sorting_fixture;
        for (int j = 0; j < pads; ++j) append_box(sorting_fixture, {0, 4. * (j - pads / 2), 0});
        auto sorted_geometry = cuda_columns(Solid(std::move(sorting_fixture)), true, .4);
        sorted_scene.geometry = {sorted_geometry, sorted_geometry};
        const auto reference = device::advance_resident(sorted_scene, dt, 2,
            device::RegionReduction::ordered, device::ResidentExecution::cached);
        const auto staged = device::advance_resident(sorted_scene, dt, 2,
            device::RegionReduction::audit, device::ResidentExecution::wavefront);
        const auto pooled = device::advance_resident(sorted_scene, dt, 2,
            device::RegionReduction::audit, device::ResidentExecution::coherent);
        require(std::equal(pooled.frames.begin(), pooled.frames.end(), reference.frames.begin(), reference.frames.end(),
                           [](double a, double b) { return a == b || (std::isnan(a) && std::isnan(b)); }),
                "bounded region pool changed separated contacts when refilling slots");
        std::cout << "candidate_sort pads=" << pads << " max_proposals=" << staged.maximum_proposals << '\n';
        require(std::equal(staged.frames.begin(), staged.frames.end(), reference.frames.begin(), reference.frames.end(),
                           [](double a, double b) { return a == b || (std::isnan(a) && std::isnan(b)); }),
                "merged/global candidate sorting changed analytic contact response");
        require(staged.maximum_proposals > 32 &&
                    (pads == 1 ? staged.maximum_proposals <= 1024 : staged.maximum_proposals > 1024),
                "candidate sorting regression did not exercise both storage paths");
    }

    // Independent contact pairs enter and leave at different times. The second
    // pair is initially open, so queue offsets must handle empty middle ranges.
    auto multiple=scene;
    multiple.coordinates.push_back({2,0,-.01,0,0,0,0,0,0});
    multiple.geometry.insert(multiple.geometry.end(),{geometry,geometry});
    multiple.bodies.push_back({{{0,.01,0}},z,zero,0,-1,-1,zero});
    multiple.bodies.push_back({{{.0021,.01,0}},z,zero,0,-1,1,x});
    multiple.pairs.push_back({2,3,2e4,1,0,.0002,true});
    const auto multiple_ref=device::advance_resident(multiple,dt,steps);
    const auto multiple_batch=device::advance_resident(multiple,dt,steps,device::RegionReduction::audit,
                                                       device::ResidentExecution::batched);
    require(std::equal(multiple_batch.frames.begin(),multiple_batch.frames.end(),multiple_ref.frames.begin(),multiple_ref.frames.end(),
                       [](double a,double b){return a==b || (std::isnan(a) && std::isnan(b));}),
            "batched cross-pair execution changed independent contact dynamics");
    const auto multiple_wave=device::advance_resident(multiple,dt,steps,device::RegionReduction::audit,
                                                       device::ResidentExecution::wavefront);
    require(std::equal(multiple_wave.frames.begin(),multiple_wave.frames.end(),multiple_ref.frames.begin(),multiple_ref.frames.end(),
                       [](double a,double b){return a==b || (std::isnan(a) && std::isnan(b));}),
            "deferred queues changed independent contact pairs");
    const auto audit=device::advance_resident(scene,dt,steps,device::RegionReduction::audit);
    require(std::equal(audit.frames.begin(),audit.frames.end(),result.frames.begin(),result.frames.end(),
                       [](double a,double b){return a==b || (std::isnan(a) && std::isnan(b));}) &&
            audit.region_audit.size()==steps+1,
            "region audit changed resident dynamics");
    unsigned checked=0,entered=0,exited=0;
    for(const auto &row:audit.region_audit) {
        checked+=row.regions;
        entered+=row.entered_faces;
        exited+=row.exited_faces;
        for(int d=0;d<3;++d)
            require(row.force_error[d]<1e-10 && row.torque_a_error[d]<1e-12 && row.torque_b_error[d]<1e-12,
                    "single-region box wrench differs from serial");
    }
    require(checked && entered && exited,"missing contact or release in region audit");
    // Two disconnected pads belong to one rigid body. Their contact witnesses
    // must remain in two regions: the segment between them crosses a pad wall.
    PartFile pads, plate;
    append_box(pads,{0,-3,0});
    append_box(pads,{0,3,0});
    append_box(plate,{}, {1,4,1});
    planar_topology(pads); planar_topology(plate);
    auto separated=scene;
    separated.geometry={cuda_columns(Solid(std::move(pads)),true,.4),
                        cuda_columns(Solid(std::move(plate)),true,.4)};
    const auto plain_pads=device::advance_resident(separated,dt,40);
    const auto cached_pads=device::advance_resident(separated,dt,40,device::RegionReduction::audit,
                                                    device::ResidentExecution::cached);
    require(cached_pads.blocker_hits>0,"blocking-face reuse was not exercised");
    const auto batch_pads=device::advance_resident(separated,dt,40,device::RegionReduction::audit,
                                                   device::ResidentExecution::batched);
    require(std::equal(batch_pads.frames.begin(),batch_pads.frames.end(),plain_pads.frames.begin(),plain_pads.frames.end(),
                       [](double a,double b){return a==b || (std::isnan(a) && std::isnan(b));}),
            "batched links changed disconnected-pad dynamics");
    const auto wave_pads=device::advance_resident(separated,dt,40,device::RegionReduction::audit,
                                                   device::ResidentExecution::wavefront);
    require(std::equal(wave_pads.frames.begin(),wave_pads.frames.end(),plain_pads.frames.begin(),plain_pads.frames.end(),
                       [](double a,double b){return a==b || (std::isnan(a) && std::isnan(b));}),
            "ordered connectivity merged disconnected pads");
    // Partition one physical plane into four charts; both schedules must
    // reconstruct the same continuous contact region and force.
    auto split=box();
    const auto original_face=split.faces[1];
    const auto original_patch=split.geometry.patches[1];
    for (int k=0;k<4;++k) {
        auto face=original_face; auto patch=original_patch;
        face.domain.u={-1+.5*k,-1+.5*(k+1)};
        face.lo[1]=face.domain.u.lo; face.hi[1]=face.domain.u.hi;
        patch.domain=face.domain;
        if (!k) { split.faces[1]=face; split.geometry.patches[1]=patch; }
        else {
            split.faces.push_back(face); patch.face_id=split.faces.size();
            split.geometry.patches.push_back(patch);
        }
    }
    auto partitioned=scene;
    planar_topology(split);
    partitioned.geometry={cuda_columns(Solid(std::move(split)),true,.4),geometry};
    const auto partitioned_ref=device::advance_resident(partitioned,dt,40,device::RegionReduction::ordered,
                                                        device::ResidentExecution::batched);
    const auto partitioned_wave=device::advance_resident(partitioned,dt,40,device::RegionReduction::audit,
                                                        device::ResidentExecution::wavefront);
    require(partitioned_wave.audited_groups>0,"partitioned charts were not audited");
    require(std::equal(partitioned_wave.frames.begin(),partitioned_wave.frames.end(),partitioned_ref.frames.begin(),partitioned_ref.frames.end(),
                       [](double a,double b){return a==b || (std::isnan(a) && std::isnan(b));}),
            "ordered links changed a partitioned physical contact region");
    require(std::equal(cached_pads.frames.begin(),cached_pads.frames.end(),plain_pads.frames.begin(),plain_pads.frames.end(),
                       [](double a,double b){return a==b || (std::isnan(a) && std::isnan(b));}),
            "blocking-face reuse changed disconnected-pad dynamics");
    unsigned long long topology_checks=0,topology_hits=0;
    for(const auto *fixture:{&multiple,&separated,&partitioned}) {
        const auto original=device::advance_resident(*fixture,dt,40,device::RegionReduction::ordered,
                                                     device::ResidentExecution::wavefront);
        for(unsigned mask:{1u,2u,4u,8u,15u,25u,57u,121u,153u}) {
            const auto improved=device::advance_resident(*fixture,dt,40,device::RegionReduction::audit,
                                                         device::ResidentExecution::coherent,false,mask);
            require(std::equal(improved.frames.begin(),improved.frames.end(),original.frames.begin(),original.frames.end(),
                              [mask](double a,double b){return a==b || (std::isnan(a) && std::isnan(b)) ||
                                  ((mask&32) && std::abs(a-b)<1e-10*(1+std::abs(b)));}),
                    "coherent ablation changed pair/region topology or analytic dynamics");
            if(mask&64) require(improved.warm_hits>0,"warm projection was not exercised");
            if(mask==57) { topology_checks+=improved.topology_checks; topology_hits+=improved.topology_hits; }
        }
    }
    require(topology_checks>0 && topology_hits>0,"shared-edge connection proposals were not exercised");
    std::cout<<"topology probes="<<topology_checks<<" connected="<<topology_hits<<'\n';
    // The same contact plus independent, analytically accelerated coordinates
    // checks active-prefix handling both below and at the native capacity.
    for (int n : {3,16}) {
        auto expanded=scene;
        for (int j=1;j<n;++j)
            expanded.coordinates.push_back({double(j+1),0,0,0,0,0,.001,0,0});
        const auto many=device::advance_resident(expanded,dt,steps,device::RegionReduction::ordered,
                                                 device::ResidentExecution::batched);
        const int stride=1+4*n+3;
        for (int i=0;i<=steps;++i) {
            const double *row=many.frames.data()+stride*i;
            for (int j=1;j<n;++j) {
                const double acceleration=.001/(j+1),t=i*dt;
                require(std::abs(row[1+4*j]-.5*acceleration*t*t)<1e-12 &&
                        std::abs(row[2+4*j]-acceleration*t)<1e-12 &&
                        std::abs(row[3+4*j]-acceleration)<1e-12,
                        "active-coordinate copy changed constant-acceleration response");
            }
            require(std::abs(row[1]-result.frames[width*i+1])<1e-8,
                    "independent coordinates changed box displacement");
        }
    }
}
#endif
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
            auto gpu=cuda_columns(solid,false,1.);
            std::vector<device::Ray> rays{
                {{{.25,.49,-.2}},{{0,0,1}}}, {{{0,.49,.01}},{{1,0,0}}},
                {{{0,.49,1e-8}},{{1,0,0}}}, {{{0,.49,-1e-6}},{{1,0,0}}},
                {{{-1,.49,-.2}},{{0,0,1}}}
            };
            std::vector<device::Hit> hits;
            gpu->trace(rays,1,hits);
            std::vector<device::Hit> fast;
            gpu->trace(rays,1,fast,true);
            if (!rational) require(fast[0].face>=0,"precomputed cubic crossing was not exercised");
            for(size_t i=0;i<fast.size();++i) if(fast[i].face!=-3) {
                require(fast[i].face==hits[i].face,"precomputed classification differs from complete query");
                if(fast[i].face>=0) require(fast[i].t==hits[i].t && fast[i].u==hits[i].u && fast[i].v==hits[i].v,
                                            "precomputed root differs from complete query");
            }
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
#ifdef NC_WITH_CUDA_COLUMNS
        if(argc>2 && std::string_view(argv[1])=="--describe-parts") {
            for(int i=2;i<argc;++i)describe_part(argv[i]);
            return 0;
        }
        if(argc>2 && std::string_view(argv[1])=="--audit-parts"){
            bool passed=true;
            for(int i=2;i<argc;++i)passed=audit_part(argv[i]) && passed;
            return passed?0:1;
        }
#endif
        bool cuda=argc==2 && std::string_view(argv[1])=="--cuda";
        require(argc==1 || cuda,"expected --cuda or no arguments");
        overlap(cuda);
        split_and_swap();
        trims();
        curved_roots(cuda);
#ifdef NC_WITH_CUDA_COLUMNS
        if (cuda) {
            revolved_rays();transverse_rays();resident_boxes();
        }
#endif
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
