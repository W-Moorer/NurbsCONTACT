#include "nurbscontact/cad_solid_contact.hpp"
#include "nurbscontact/cad_contact_regions.hpp"
#include "../src/cad_boundary_distance.hpp"
#include "../src/cad_bezier_ray.hpp"
#include "../src/cad_column_depth.hpp"
#include "../src/cad_compiled_chart.hpp"
#include "../src/cad_span_index.hpp"
#include "../src/cad_patch_search.hpp"
#include "../src/cad_local_solid.hpp"
#include "../src/cad_planar_trim.hpp"
#include "../src/cad_rigid_path.hpp"
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBuilderAPI_NurbsConvert.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <Geom_BezierSurface.hxx>
#include <GeomConvert.hxx>
#include <BRep_Builder.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS_Compound.hxx>
#include <iostream>

using namespace nc;
using namespace nc::dynamics;

namespace {
void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
void close(double actual,double expected,double tol,const char* what) {
    if (std::abs(actual-expected)<=tol) return;
    std::cerr<<what<<" actual "<<actual<<", expected "<<expected<<"\n";
    throw std::runtime_error(what);
}

void bezier_rays() {
    // Bilinear z=.1*u*v embedded in a cubic control net. The two finite
    // rays have different directions and cross at independently known points.
    cad::ColumnDepthBound::Net net;
    for (int i=0;i<4;++i) for (int j=0;j<4;++j) net[i][j]={i/3.,j/3.,.1*i*j/9.};
    const cad::ColumnDepthBound bound(net);
    const gp_XYZ v0=gp_Dir(.03,.01,1).XYZ(),v1=gp_Dir(-.02,.04,1).XYZ();
    std::array<cad::MaterialRay,2> packet{{{gp_XYZ(.3,.45,.0135)-v0*.25,v0},
                                          {gp_XYZ(.6,.55,.033)-v1*.27,v1}}};
    using State=cad::ColumnDepthBound::State;
    std::array projected{bound.project(packet[0]),bound.project(packet[1])};
    require(bound.classify(projected,.6,1)==State::interior,"known warped packet lost");
    require(bound.classify(projected,.05,1)==State::empty,"finite ray missed its depth limit");
    require(bound.classify(projected,.6,-1)!=State::interior,"entry accepted as a solid exit");
    TColgp_Array2OfPnt cubic(1,4,1,4);
    for (int i=0;i<4;++i) for (int j=0;j<4;++j) cubic(i+1,j+1)=gp_Pnt(net[i][j]);
    auto support=GeomConvert::SurfaceToBSplineSurface(new Geom_BezierSurface(cubic));
    cad::CompiledChart full(BRepBuilderAPI_MakeFace(support,1e-7).Face(),true,true);
    cad::CompiledChart cut_patch(BRepBuilderAPI_MakeFace(support,.4,.8,.2,.8,1e-7).Face(),true,true);
    require(full.depth_bounds.size()==1 && cut_patch.depth_bounds.size()==1,"missing polynomial depth bounds");
    double depth;
    require(full.interior_depth(0,packet[0],.6,depth),"lost interior curved root");
    close(depth,.25,1e-10,"curved depth refinement");
    require(!cut_patch.interior_depth(0,packet[0],.6,depth),"interior root outside real trim");
    require(cut_patch.interior_depth(0,packet[1],.6,depth),"lost root inside trimmed support");
    close(depth,.27,1e-10,"second curved depth");
    packet[0].point.SetX(-.5);
    projected[0]=bound.project(packet[0]);
    require(bound.classify(projected,.6,1)==State::uncertain,"packet straddling an edge was not split");
    // No assumption about spatial order of consecutive knot spans. Each leaf
    // may include extra candidates; it must preserve all overlapping spans.
    struct TestBox {gp_XYZ lo,hi;};
    std::vector<TestBox> boxes;
    for (int j=0;j<52;++j) {
        const double x=(j*7)%52;
        boxes.push_back({{x,-1,-1},{x+1,1,1}});
    }
    cad::SpanIndex spans;
    spans.build(boxes.size(),[&](auto i)->const TestBox& {return boxes[i];});
    for (double x:{0.,7.25,29.,51.5,65.}) {
        std::vector<std::size_t> expected,visited;
        for (std::size_t j=0;j<boxes.size();++j)
            if (boxes[j].lo.X()<=x && x<=boxes[j].hi.X()) expected.push_back(j);
        spans.query({x,0,0},{x,0,0},[&](auto j){
            if (boxes[j].lo.X()<=x && x<=boxes[j].hi.X()) visited.push_back(j);
            return true;
        });
        require(visited==expected,"span hierarchy changed root-test coverage or order");
    }
    TColgp_Array2OfPnt poles(1,3,1,3);
    TColStd_Array2OfReal weights(1,3,1,3);
    for (int i=1;i<=3;++i) for (int j=1;j<=3;++j) {
        poles(i,j)=gp_Pnt(i-1,j-1,.12*(i-1)*(j-1));
        weights(i,j)=1+.015*(i+j);
    }
    Handle(Geom_BezierSurface) surface=new Geom_BezierSurface(poles,weights);
    const auto face=BRepBuilderAPI_MakeFace(surface,1.e-7).Face();
    cad::PatchAssets assets(face);
    auto atlas=std::make_shared<cad::BezierRayAtlas>(face,assets);
    CadContactIndex index(face);
    CadExitQuery old(index),fast(index,atlas);
    const gp_XYZ dir=gp_Dir(.2,-.1,1).XYZ();
    for (int i=1;i<=5;++i) for (int j=1;j<=5;++j) {
        gp_Pnt q; gp_Vec a,b;
        surface->D1(i/6.,j/6.,q,a,b);
        const auto p=q.XYZ()-dir*.1;
        Vec origin{p.X()*.001,p.Y()*.001,p.Z()*.001},d{dir.X(),dir.Y(),dir.Z()};
        std::size_t count=0;
        const auto reference=old.first_exit(origin,d,.0002,0,count);
        const auto hit=fast.first_exit(origin,d,.0002,0,count);
        require(reference.face==1 && hit.face==1,"rational curved face ray");
        close(hit.distance_m,reference.distance_m,1e-10,"Bezier exit distance");
        close(length(sub(reference.outward,hit.outward)),0.,1e-7,"rational Bezier normal");
    }
    require(fast.ray_counts()[1]>0,"rational test did not exercise the compiled intersection");
    // Preparing another packet must not eliminate hits outside its old box.
    fast.prepare_columns({.1,.1,.1},{.101,.101,.101});
    const auto mid=surface->Value(.5,.5).XYZ()-dir*.1;
    const Vec source{mid.X()*.001,mid.Y()*.001,mid.Z()*.001},direction{dir.X(),dir.Y(),dir.Z()};
    std::size_t packet_tests=0;
    const std::array<std::size_t,1> required{1};
    const auto fresh=old.first_exit(source,direction,.0002,1,packet_tests,required);
    const auto outside_packet=fast.first_exit(source,direction,.0002,1,packet_tests,required);
    require(fresh.face==1 && outside_packet.face==1,"stale packet lost a valid ray");
    close(fresh.distance_m,outside_packet.distance_m,1e-10,"packet coverage after pose change");
    gp_Trsf advance;
    advance.SetTranslation(gp_Vec(dir)*(-.04));
    BRep_Builder assembly;
    TopoDS_Compound two_faces;
    assembly.MakeCompound(two_faces);
    assembly.Add(two_faces,face);
    assembly.Add(two_faces,face.Moved(TopLoc_Location(advance)));
    CadContactIndex overlap_index(two_faces);
    CadExitQuery occluded(overlap_index);
    for (double box_start:{.1,-.005}) {
        occluded.prepare_columns({box_start,box_start,box_start},{box_start+.01,box_start+.01,box_start+.01});
        require(!occluded.first_exit(source,direction,.0002,1,packet_tests,required).face,
                "packet ignored a closer face outside the load group");
    }
    std::optional<cad::BezierRayHit> interior;
    require(atlas->query(1,{100,100,100},dir,1,1e-7,interior) && !interior,"separated Bezier ray");
    // A root at the UV boundary must go back to the shared-edge classifier.
    const auto corner=surface->Value(0,0).XYZ()-dir*.1;
    require(!atlas->query(1,corner,dir,.2,1e-7,interior),"Bezier corner treated as an isolated interior root");

    // The supporting surface also exists outside the trimmed UV rectangle.
    const auto trimmed=BRepBuilderAPI_MakeFace(surface,.25,.75,.25,.75,1e-7).Face();
    cad::PatchAssets cut_assets(trimmed);
    CadContactIndex cut_index(trimmed);
    CadExitQuery cut(cut_index,std::make_shared<cad::BezierRayAtlas>(trimmed,cut_assets));
    const auto outside=surface->Value(.1,.5).XYZ()-dir*.1;
    std::size_t tests=0;
    require(!cut.first_exit({outside.X()*.001,outside.Y()*.001,outside.Z()*.001},
                            {dir.X(),dir.Y(),dir.Z()},.0002,0,tests).face,"ray entered removed CAD trim");

    // Fold x=3(u-.5)^2, y=v, z=u. A vertical line can meet a single span twice.
    for (int i=1;i<=3;++i) for (int j=1;j<=3;++j)
        poles(i,j)=gp_Pnt(i==2 ? -.75 : .75,(j-1)*.5,(i-1)*.5);
    const auto folded=BRepBuilderAPI_MakeFace(new Geom_BezierSurface(poles),1e-7).Face();
    cad::PatchAssets folded_assets(folded);
    cad::BezierRayAtlas fold(folded,folded_assets);
    for (double x:{0.,.1875}) {
        std::optional<cad::BezierRayHit> ambiguous;
        require(!fold.query(1,{x,.4,-.1},{0,0,1},1.2,1e-7,ambiguous),"tangent or double root accepted");
    }
}
void local_region_depths() {
    const auto ball=BRepPrimAPI_MakeSphere(10.).Shape();
    const ContactPose origin{},right{{.019,0,0}};
    ContactWork work;
    const auto query=cad_local_region_query(ball,ball,{},.002);
    const auto hit=query->regions(origin,right,work),swapped=query->regions(right,origin,work);
    require(hit.size()==1 && swapped.size()==1,"local sphere region count");
    close(hit[0].depth_m,.001,1e-10,"analytic local sphere depth");
    close(length(sub(hit[0].point,swapped[0].point)),0,1e-10,"swapped force arm");
    close(length(add(hit[0].normal,swapped[0].normal)),0,1e-10,"swapped normal");

    const auto box=BRepPrimAPI_MakeBox(10,10,10).Shape();
    const auto split=BRepAlgoAPI_Fuse(BRepPrimAPI_MakeBox(10,5,10).Shape(),
        BRepPrimAPI_MakeBox(gp_Pnt(0,5,0),10,5,10).Shape()).Shape();
    const ContactPose shift{{.009,0,0}};
    const auto plain=cad_local_region_query(box,box,{},.002)->regions(origin,shift,work);
    const auto divided=cad_local_region_query(split,box,{},.002)->regions(origin,shift,work);
    require(plain.size()==1 && divided.size()==1,"CAD face split duplicated local force");
    close(divided[0].depth_m,plain[0].depth_m,1e-10,"split depth");
    close(length(sub(divided[0].point,plain[0].point)),0,1e-10,"split force arm");

    CadExitQuery ray{CadContactIndex(box)};
    std::size_t tests=0;
    require(!ray.first_exit({-.0001,.005,.005},{1,0,0},.012,0,tests).face,"outside entry accepted as an exit");
    require(!ray.first_exit({.005,.005,.005},{1,0,0},.001,0,tests).face,"exit beyond reach");
    const Vec corner=scale(Vec{1,1,0},1/std::sqrt(2.));
    require(!ray.first_exit({.005,.005,.005},corner,.01,0,tests).face,"sharp edge accepted as smooth");
    const auto hole=BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(5,5,-1),gp_Dir(0,0,1)),2,12).Shape();
    CadExitQuery plate{CadContactIndex(BRepAlgoAPI_Cut(box,hole).Shape())};
    require(!plate.interior_segment({.001,.005,.005},{.009,.005,.005}),"connected through a hole");
    require(plate.interior_segment({.001,.004,.005},{.001,.006,.005}),"interior path rejected");

    BRep_Builder builder;
    TopoDS_Compound first,second;
    builder.MakeCompound(first); builder.MakeCompound(second);
    builder.Add(first,ball); builder.Add(second,ball);
    builder.Add(first,BRepPrimAPI_MakeSphere(gp_Pnt(0,35,0),10).Shape());
    builder.Add(second,BRepPrimAPI_MakeSphere(gp_Pnt(.6,35,0),10).Shape());
    const auto two=cad_local_region_query(first,second,{},.002);
    require(two->regions(origin,ContactPose{{.0195,0,0}},work).size()==1,"spurious second contact");
    require(two->regions(origin,right,work).size()==2,"missed new contact while the first persisted");
}

void pressure_moment() {
    // Every pressure normal through the spherical patch passes through the
    // sphere center. Two unequal cuts give a noncentral volume centroid.
    const auto ball=BRepPrimAPI_MakeSphere(10.).Shape();
    const auto corner=BRepPrimAPI_MakeBox(gp_Pnt(6.,2.,-12.),gp_Pnt(12.,12.,12.)).Shape();
    for (bool swap:{false,true}) {
        ContactWork work;
        std::vector<Vec> arms;
        const auto regions=cad::solid_regions(swap ? corner:ball,swap ? ball:corner,work,false,&arms);
        require(regions.size()==1 && arms.size()==1,"sphere pressure patch missing");
        require(length(arms[0])>1e-5,"asymmetric sphere pressure patch needs a couple");
        close(length(add(cross3(regions[0].point,regions[0].normal),arms[0])),0.,2e-8,
              "uniform spherical pressure must have zero moment about sphere center");
    }
    const auto patch=cad_patch_region_query(ball,ball,{},.002);
    ContactWork work;
    const auto closed=patch->regions({},ContactPose{{0,0,.019}},work);
    require(closed.size()==1,"empty chart candidates lost sphere pole contact");
    close(closed[0].depth_m,.001,1e-8,"sphere pole pressure depth");
    const auto open=patch->regions({},ContactPose{{0,0,.021}},work);
    require(open.empty(),"separated spheres acquired pressure");

    cad::PatchSearch search(ball,ball,6,.003,true);
    const std::array poses{ContactPose{},ContactPose{{.018,0,0}}};
    const auto first=search.chord(poses,{.009,0,0},{1,0,0},{});
    const auto second=search.chord(poses,{.009,.001,0},{.96,.28,0},{});
    require(first && second,"expected two interior sphere chords");
    const auto x=search.integrate(poses,{*first},0);
    const auto y=search.integrate(poses,{*second},0);
    require(x && y,"sphere pressure integration failed");
    close(length(sub(x->point,y->point)),0,1e-12,"pressure centroid depends on witness");
    close(length(sub(x->pressure_arm,y->pressure_arm)),0,1e-12,"pressure moment depends on witness");
    close(std::abs(x->depth_m-y->depth_m),0,1e-12,"pressure depth depends on witness");
    // The first moment of a material block must include the translation of
    // each body when the two sets of columns are combined.
    const ContactPose frame{{.073,-.051,.034},{0,0,1},.63};
    const std::array moved{frame,ContactPose{frame.point(poses[1].origin),frame.axis,frame.angle}};
    const auto witness=search.chord(moved,frame.point({.009,0,0}),frame.rotate({1,0,0}),{});
    require(witness.has_value(),"moved pressure witness missing");
    const auto rotated=search.integrate(moved,{*witness},0);
    require(rotated.has_value(),"moved pressure block unresolved");
    close(length(sub(rotated->point,frame.point(x->point))),0,2e-10,"rigid transformation of pressure centroid");
    close(length(sub(rotated->normal,frame.rotate(x->normal))),0,2e-8,"rigid transformation of pressure resultant");
    close(length(sub(rotated->pressure_arm,frame.rotate(x->pressure_arm))),0,2e-10,"rigid transformation of pressure couple");
    const auto initial=search.discover(poses,true,true);
    search.discover({poses[0],ContactPose{{0,.018,0}}},true,true);
    const auto revisit=search.discover(poses,true,true);
    require(initial.window_builds==2 && revisit.window_builds==0,"did not reuse recurring relative placement");
    require(initial.candidate_pairs==revisit.candidate_pairs && initial.labels==revisit.labels,
            "revisited sphere contact candidates or component labels changed");

    BRep_Builder builder;
    TopoDS_Compound separated;
    builder.MakeCompound(separated);
    builder.Add(separated,ball);
    builder.Add(separated,BRepPrimAPI_MakeSphere(gp_Pnt(0,25,0),10).Shape());
    CadContactIndex both(separated);
    CadExitQuery ray(both);
    std::size_t tests=0;
    const Vec direction{0,1,0};
    const auto front_exit=ray.first_exit({},direction,.05,0,tests);
    const auto back_exit=ray.first_exit({0,.025,0},direction,.05,0,tests);
    require(front_exit.face && back_exit.face && front_exit.face!=back_exit.face,"expected two distinct exit faces");
    const std::array allowed_near{front_exit.face},allowed_far{back_exit.face};
    require(ray.first_exit({},direction,.05,front_exit.face,tests,allowed_near).face==front_exit.face,
            "filtered query lost the first exit");
    require(!ray.first_exit({},direction,.05,back_exit.face,tests,allowed_far).face,
            "filtered query accepted an occluded exit");
}

void check(CadEventMethod chen) {
    const auto ball=BRepPrimAPI_MakeSphere(10.).Shape();
    auto balls=cad_solid_region_query(ball,ball,{},chen);
    ContactWork work;
    const ContactPose origin{},positive{{.019,0,0}},negative{{-.019,0,0}};
    for (auto pose:{positive,negative}) {
        const auto r=balls->regions(origin,pose,work);
        require(r.size()==1,"sphere lens requires one force");
        close(r[0].depth_m,.001,1e-8,"sphere depth");
        close(r[0].point[0],pose.origin[0]/2,1e-8,"sphere contact point");
        close(r[0].normal[0],std::copysign(1.,pose.origin[0]),1e-8,"sphere normal");
    }
    const ContactPose start{{-.035,0,0}},end{{.035,0,0}};
    const auto sweep=balls->transition(origin,start,origin,end,1e-8);
    require(!sweep.start_in_contact && !sweep.end_in_contact && sweep.fraction,"step-interior sphere hit");
    close(*sweep.fraction,15./70.,5e-7,"sphere entry time");
    require(chen==CadEventMethod::conservative_advancement || sweep.chen_queries>0,"expected calls to Chen patch CCD");
    require(chen!=CadEventMethod::chen_2026 || sweep.tdib_cells>0,"expected safeguarded envelope search");
    const auto release=balls->transition(origin,positive,origin,end,1e-8);
    require(release.start_in_contact && !release.end_in_contact && release.fraction,"sphere release");
    close(*release.fraction,1./16.,1e-6,"sphere release time");

    const auto box=BRepPrimAPI_MakeBox(10.,10.,10.).Shape();
    const auto blocks=cad_solid_region_query(box,box,{},chen);
    if (chen==CadEventMethod::chen_2026) {
        const auto crossing=blocks->transition(origin,ContactPose{{-.020,0,0}},origin,ContactPose{{.020,0,0}},1e-8);
        require(crossing.fraction && crossing.chen_confirmed>0,"expected a trimmed-solid-confirmed planar candidate");
        close(*crossing.fraction,.25,1e-6,"planar CCD proposal");
    }
    const auto overlap=blocks->regions(origin,ContactPose{{.009,0,0}},work);
    require(overlap.size()==1,"coplanar box boundaries");
    close(overlap[0].normal[0],1.,1e-8,"box normal");
    close(overlap[0].depth_m,.001,1e-8,"box depth");
    const auto booleans=work.cad_boolean_queries;
    const auto cached=blocks->regions(origin,ContactPose{{.009,0,0}},work);
    require(cached.size()==1 && cached[0].point==overlap[0].point && work.cad_boolean_queries==booleans
            && work.cad_region_cache_hits>0,"identical poses did not reuse the same CAD region");

    const auto half=BRepPrimAPI_MakeBox(10.,5.,10.).Shape();
    const auto other_half=BRepPrimAPI_MakeBox(gp_Pnt(0,5,0),10.,5.,10.).Shape();
    const auto divided=BRepAlgoAPI_Fuse(half,other_half).Shape();
    int face_count=0;
    for (TopExp_Explorer it(divided,TopAbs_FACE);it.More();it.Next()) ++face_count;
    require(face_count>6,"expected a coplanar split on the loaded face");
    const auto split=cad_solid_region_query(divided,box,{},chen)->regions(origin,ContactPose{{.009,0,0}},work);
    require(split.size()==1,"splitting the CAD face duplicated the load");
    close(length(sub(split[0].point,overlap[0].point)),0.,1e-8,"split contact point");
    close(length(sub(split[0].normal,overlap[0].normal)),0.,1e-8,"split contact normal");
    close(split[0].depth_m,overlap[0].depth_m,1e-8,"split contact depth");

    BRep_Builder builder;
    TopoDS_Compound cluster;
    builder.MakeCompound(cluster);
    builder.Add(cluster,ball);
    builder.Add(cluster,BRepPrimAPI_MakeSphere(gp_Pnt(0,35,0),10.).Shape());
    const auto patches=cad_solid_region_query(cluster,cluster,{},chen)->regions(origin,positive,work);
    require(patches.size()==2,"disconnected loaded regions must retain independent forces");
    for (const auto& region:patches) close(region.depth_m,.001,1e-8,"multi-region depth");

    const auto disk=BRepPrimAPI_MakeCylinder(10.,2.).Shape();
    const auto tip=BRepPrimAPI_MakeSphere(1.).Shape();
    const auto truncated=cad_solid_region_query(disk,tip,{},chen);
    const auto miss=truncated->transition(origin,ContactPose{{.009,.009,-.003}},origin,ContactPose{{.009,.009,.005}},1e-8);
    require(!miss.fraction,"motion outside circular trim produced a contact");
    const auto far_away=balls->transition(origin,ContactPose{{1,0,0}},origin,ContactPose{{2,0,0}},1e-8);
    require(!far_away.fraction && far_away.cad_distances==0,"separated swept boxes reached the CAD kernel");

    // Two unit-radius spheres. The second rotates about a local pivot at
    // (10,0,0) mm. Its center traverses a semicircle, meeting the first midway.
    const ContactPose fixed{{.010,-.010,0}},rotated{{.020,0,0},{0,0,1},std::acos(-1.)};
    const auto hinged=cad_solid_region_query(tip,tip,{Vec{},Vec{.010,0,0}},chen);
    const auto hit=hinged->transition(fixed,origin,fixed,rotated,1e-8);
    require(hit.fraction && !hit.start_in_contact && !hit.end_in_contact,"offset-pivot arc missed sphere");
    close(*hit.fraction,.5-2*std::asin(.1)/std::acos(-1.),2e-6,"rotating sphere entry time");
}

void planar_trim_bounds() {
    const auto disk=BRepPrimAPI_MakeCylinder(10.,2.).Shape();
    const auto ring=BRepAlgoAPI_Cut(disk,BRepPrimAPI_MakeCylinder(5.,2.).Shape()).Shape();
    bool checked=false;
    for (TopExp_Explorer it(ring,TopAbs_FACE);it.More();it.Next()) {
        const auto face=TopoDS::Face(it.Current());
        BRepAdaptor_Surface surf(face);
        if (surf.GetType()!=GeomAbs_Plane || std::abs(surf.Plane().Location().Z()-2.)>1e-8) continue;
        cad::PlanarTrim trim(face,surf.Plane());
        auto excluded=[&](std::initializer_list<gp_XYZ> pts){return trim.separates(pts,gp_Trsf(),1e-5);};
        require(excluded({{0.,0.,2.}}),"failed to exclude the inner trim of an annulus");
        require(!excluded({{5.,0.,2.}}),"rejected inner rim tangency");
        require(!excluded({{7.,0.,2.}}),"rejected annular material");
        require(!excluded({{0.,0.,2.},{12.,0.,2.}}),"missed segment bridging both trims");
        require(!excluded({{0.,0.,1.},{14.,0.,3.}}),"missed an oblique segment through the face");
        require(excluded({{0.,0.,2.},{7.,0.,5.}}),"failed to restrict the projection to the plane slab");
        for (int i=0;i<72;++i) {
            const double angle=i*std::acos(-1.)/36.;
            for (double radius:{5.,5.000003,7.5,9.999997,10.}) {
                require(!excluded({{radius*std::cos(angle),radius*std::sin(angle),2.}}),
                        "winding or its tolerance rejected material on the annular face");
            }
        }
        gp_Trsf turn;
        turn.SetRotation(gp_Ax1({},gp_Dir(1.,2.,3.)),.73);
        turn.SetTranslationPart(gp_Vec(8.,-4.,12.));
        const auto moved=TopoDS::Face(face.Moved(TopLoc_Location(turn)));
        cad::PlanarTrim rotated(moved,BRepAdaptor_Surface(moved).Plane());
        require(rotated.separates({{0.,0.,2.}},turn,1e-5) &&
                !rotated.separates({{7.,0.,2.}},turn,1e-5),"rotated face changed its winding");
        checked=true;
    }
    require(checked,"annular test face missing");
}

void boundary_distances() {
    const auto box=BRepPrimAPI_MakeBox(10.,10.,10.).Shape();
    const auto ball=BRepPrimAPI_MakeSphere(1.).Shape();
    const auto a=cad::boundary_index(ball),b=cad::boundary_index(box);
    SurfaceTransition counts;
    const auto check=[&](gp_XYZ center,double cap,double expected) {
        gp_Trsf transform;
        transform.SetTranslationPart(gp_Vec(center));
        const auto d=cad::bounded_distance(a,b,transform,cap,1e-7,counts);
        close(d,expected,2e-6,"bounded feature distance");
        BRepExtrema_DistShapeShape ref(ball.Moved(TopLoc_Location(transform)),box,1e-7);
        require(ref.IsDone(),"whole-solid reference failed");
        close(d,std::min(cap,ref.Value()),2e-6,"reference distance mismatch");
    };
    check({14,5,5},10,3);
    check({14,5,5},.5,.5);
    check({11,5,5},10,0);
    check({10,5,5},10,0);
    check({5,5,5},10,0);
    require(counts.cad_extrema && counts.cad_classifications,"missing feature or containment checks");
    // Flat spline faces can span an entire opposing planar trim. Their trims
    // must still recover the distance for parallel, tilted and crossing poses.
    const auto plate=BRepBuilderAPI_NurbsConvert(BRepPrimAPI_MakeBox(20.,2.,1.).Shape()).Shape();
    const auto plate_index=cad::boundary_index(plate);
    cad::BoundaryWitness plate_witness;
    for (double tilt:{0.,.013,-.25,.15,1.40}) {
        gp_Trsf pose;
        pose.SetRotation(gp_Ax1({},gp_Dir(0,1,0)),tilt);
        pose.SetTranslationPart(gp_Vec(-5.,4.,tilt<0. ? 7. : 11.));
        BRepExtrema_DistShapeShape ref(plate.Moved(TopLoc_Location(pose)),box,1e-7);
        require(ref.IsDone(),"plate reference failed");
        close(cad::bounded_distance(plate_index,b,pose,5.,1e-7,counts),ref.Value(),2e-6,
              "trimmed spline against planar face");
        close(cad::bounded_distance(plate_index,b,pose,5.,1e-7,counts,&plate_witness),ref.Value(),2e-6,
              "witness-aligned boxes after relative rotation");
    }
    const auto rational_ball=cad::boundary_index(BRepBuilderAPI_NurbsConvert(BRepPrimAPI_MakeSphere(1.).Shape()).Shape());
    for (double separation:{3.,2.,1.,0.}) {
        gp_Trsf pose;
        pose.SetRotation(gp_Ax1(gp_Pnt(),gp_Dir(1,2,3)),.7);
        pose.SetTranslationPart(gp_Vec(separation/std::sqrt(2.),separation/std::sqrt(2.),0));
        close(cad::bounded_distance(a,a,pose,5,1e-7,counts),std::max(0.,separation-2.),2e-6,
              "oblique smooth boundaries");
        close(cad::bounded_distance(rational_ball,rational_ball,pose,5,1e-7,counts),
              std::max(0.,separation-2.),2e-6,"positive-weight rational sphere distance");
    }
    const auto hollow=BRepAlgoAPI_Cut(BRepPrimAPI_MakeSphere(10.).Shape(),
                                      BRepPrimAPI_MakeSphere(6.).Shape()).Shape();
    const auto shell=cad::boundary_index(hollow);
    close(cad::bounded_distance(a,shell,gp_Trsf(),10,1e-7,counts),5.,2e-6,"sphere in cavity");

    BRep_Builder builder;
    TopoDS_Compound compound;
    builder.MakeCompound(compound);
    builder.Add(compound,box);
    builder.Add(compound,BRepPrimAPI_MakeBox(gp_Pnt(0,25,0),10.,10.,10.).Shape());
    builder.Add(compound,BRepPrimAPI_MakeBox(gp_Pnt(0,50,0),10.,10.,10.).Shape());
    const auto many=cad_solid_region_query(compound,box,{},false);
    const ContactPose origin{},left{{-.020,0,0}},right{{.020,0,0}};
    const auto interior=many->transition(origin,left,origin,right,1e-8,.02);
    require(interior.fraction && interior.cad_extrema>0,"indexed sweep lost an interior-only collision");
    close(*interior.fraction,.25,.02,"interior-only indexed entry");
    const auto loaded=many->transition(origin,left,origin,origin,1e-8,.02);
    require(loaded.fraction && loaded.end_in_contact,"indexed sweep lost a loaded endpoint");
    close(*loaded.fraction,.5,.02,"time-bracketed indexed entry");
    const auto repeated=many->transition(origin,left,origin,origin,1e-8,.02);
    require(repeated.fraction==loaded.fraction && repeated.cad_distance_cache_hits>=2,"indexed distance cache");

    // Translate across two disconnected solids, then replace the target index
    // as a new local cut would. A valid hint may cease to be the global winner.
    const auto compound_index=cad::boundary_index(compound);
    cad::BoundaryWitness witness;
    SurfaceTransition warm;
    for (double y:{4.,4.1,29.,54.,26.,4.}) {
        gp_Trsf pose;
        pose.SetTranslationPart(gp_Vec(y==26. ? 10.5 : 14.,y,5.));
        const double expected=cad::bounded_distance(a,compound_index,pose,60.,1e-7,counts);
        close(cad::bounded_distance(a,compound_index,pose,60.,1e-7,warm,&witness),expected,2e-6,
              "warm feature failed at component switch or overlap");
    }
    require(warm.cad_witness_improvements>0,"repeated feature never tightened the initial cap");
    const auto replaced=cad::boundary_index(box);
    gp_Trsf pose;
    pose.SetTranslationPart(gp_Vec(14,4,5));
    const auto attempts=warm.cad_witness_tests;
    close(cad::bounded_distance(a,replaced,pose,.5,1e-7,warm,&witness),.5,1e-12,"capped search after recut");
    require(warm.cad_witness_tests==attempts,"used a feature index from an old cut");
}

void local_solids() {
    using namespace cad;
    const auto a=BRepPrimAPI_MakeBox(40.,20.,15.).Shape();
    const auto b=BRepPrimAPI_MakeBox(4.,6.,3.).Shape();
    LocalSolidSearch search(a,b);
    ContactWork work;
    TopoDS_Shape first_clip;
    for (double x:{39.,39.2,-2.,8.,50.,39.}) {
        gp_Trsf t;
        t.SetTranslationPart(gp_Vec(x,3,4));
        const auto pair=search.select(t.Inverted(),.1);
        const auto placed=b.Moved(TopLoc_Location(t));
        if (x==50.) { require(pair.separated,"envelopes failed to separate distant boxes"); continue; }
        require(!pair.separated && !pair.a.shape.IsSame(a),"expected retained local CAD solid");
        if (x==39.) {
            if (first_clip.IsNull()) first_clip=pair.a.shape;
            else require(first_clip.IsSame(pair.a.shape),"returning pose rebuilt a retained pressure clip");
        }
        SurfaceTransition stats;
        close(bounded_distance(pair.a.boundary,pair.b.boundary,t.Inverted(),.1,1e-7,stats),0,1e-8,
              "cropped solids lost containment or crossing");
        if (x==8.) continue; // complete containment has no unique force normal
        std::vector<Vec> full_arms,clipped_arms;
        const auto full=solid_regions(a,placed,work,true,&full_arms);
        const auto cropped=solid_regions(pair.a.shape,pair.b.shape.Moved(TopLoc_Location(t)),work,true,&clipped_arms,true);
        require(full.size()==1 && cropped.size()==1,"cropping changed the number of regions");
        close(length(sub(full[0].point,cropped[0].point)),0,1e-10,"local force point");
        close(length(sub(full[0].normal,cropped[0].normal)),0,1e-10,"local force direction");
        close(full[0].depth_m,cropped[0].depth_m,1e-10,"local depth");
        close(length(sub(full_arms.at(0),clipped_arms.at(0))),0,1e-10,"local pressure moment arm");
    }
    // Two different off-center solids undergoing the same finite rigid motion.
    ContactPose p{},q{{.004,.005,.001},{1,2,3},.49};
    RigidPath first(p,q,{}),second(p,q,{});
    require(search.motion_bound(first,second,100.,100.)<1e-5,"common rigid motion failed to cancel");

    RigidPath moving_a(p,ContactPose{{.001,0,0},{0,0,1},.08},{.025,.009,.005});
    RigidPath moving_b(ContactPose{{.043,.003,.004}},ContactPose{{.039,.005,.004},{1,2,0},-.11},
                       {.002,.003,.001});
    const double speed=search.motion_bound(moving_a,moving_b,30.,30.);
    double previous=0;
    for (int i=0;i<=16;++i) {
        const double t=i/16.;
        BRepExtrema_DistShapeShape whole(a.Moved(TopLoc_Location(moving_a.at(t))),
                                         b.Moved(TopLoc_Location(moving_b.at(t))),1e-7);
        require(whole.IsDone(),"rotating box reference failed");
        const double d=whole.Value();
        if (i) require(16*std::abs(d-previous)<=speed+2e-6,"local motion bound below measured distance rate");
        previous=d;
    }
}
}
int main(int argc,char** argv) {
    try { if (argc==1) { bezier_rays(); local_region_depths(); pressure_moment(); planar_trim_bounds(); boundary_distances(); local_solids(); }
          check(argc==1 ? CadEventMethod::conservative_advancement :
                std::string_view(argv[1])=="--chen-2026" ? CadEventMethod::chen_2026 : CadEventMethod::chen_2024); }
    catch (const Standard_Failure& e) { std::cerr<<e.GetMessageString()<<'\n'; return 1; }
    catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
