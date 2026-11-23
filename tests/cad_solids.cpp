#include "nurbscontact/cad_solid_contact.hpp"
#include "../src/cad_boundary_distance.hpp"
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
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

void boundary_distances() {
    const auto box=BRepPrimAPI_MakeBox(10.,10.,10.).Shape();
    const auto ball=BRepPrimAPI_MakeSphere(1.).Shape();
    const auto a=cad::boundary_index(ball),b=cad::boundary_index(box);
    SurfaceTransition counts;
    const auto check=[&](gp_XYZ center,double cap,double expected) {
        gp_Trsf transform;
        transform.SetTranslationPart(gp_Vec(center));
        const auto d=cad::bounded_distance(*a,*b,transform,cap,1e-7,counts);
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
    for (double separation:{3.,2.,1.,0.}) {
        gp_Trsf pose;
        pose.SetRotation(gp_Ax1(gp_Pnt(),gp_Dir(1,2,3)),.7);
        pose.SetTranslationPart(gp_Vec(separation/std::sqrt(2.),separation/std::sqrt(2.),0));
        close(cad::bounded_distance(*a,*a,pose,5,1e-7,counts),std::max(0.,separation-2.),2e-6,
              "oblique smooth boundaries");
    }
    const auto hollow=BRepAlgoAPI_Cut(BRepPrimAPI_MakeSphere(10.).Shape(),
                                      BRepPrimAPI_MakeSphere(6.).Shape()).Shape();
    const auto shell=cad::boundary_index(hollow);
    close(cad::bounded_distance(*a,*shell,gp_Trsf(),10,1e-7,counts),5.,2e-6,"sphere in cavity");

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
}
}
int main(int argc,char** argv) {
    try { if (argc==1) boundary_distances();
          check(argc==1 ? CadEventMethod::conservative_advancement :
                std::string_view(argv[1])=="--chen-2026" ? CadEventMethod::chen_2026 : CadEventMethod::chen_2024); }
    catch (const Standard_Failure& e) { std::cerr<<e.GetMessageString()<<'\n'; return 1; }
    catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
