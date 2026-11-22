#include "nurbscontact/occt.hpp"
#include "nurbscontact/occt_contact.hpp"
#include "nurbscontact/query.hpp"

#include <BRepAlgoAPI_Cut.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <Geom_BSplineSurface.hxx>
#include <STEPControl_Writer.hxx>
#include <TColgp_Array2OfPnt.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <TColStd_Array2OfReal.hxx>
#include <gp_Ax2.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>

#include <iostream>

namespace {

using namespace nc;
namespace fs = std::filesystem;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void write_step(const TopoDS_Shape& shape, const fs::path& path) {
    STEPControl_Writer writer;
    require(writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone, "cannot transfer fixture to STEP");
    require(writer.Write(path.string().c_str()) == IFSelect_RetDone, "cannot write STEP fixture");
}

void check_directional_contact(const TopoDS_Shape& a, const TopoDS_Shape& b) {
    using namespace nc::dynamics;
    for (double mesh:{0.,.000005}) {
        const auto lower=contact_surface(a,mesh),upper=contact_surface(b,mesh);
        const Vector velocity{1.,-.2};
        std::vector<ContactBodyState> bodies{{lower.get(),{},std::vector<Twist>(2)},
            {upper.get(),{{0,0,-.0006}}, {{{1,0,0},{}},{{0,0,1},{}}}}};
        ContactPair pair{0,1,{},{},ContactDirection::world,{0,0,1},{1000,1,1,.25,.001,.0002}};
        Forces force{Vector(2),Vector(4),Vector(4)};
        ContactWork work;
        std::vector<ContactSample> samples;
        const std::span pairs(&pair,1);
        require(add_contact_forces(&force,bodies,pairs,velocity,work,&samples),"contact force did not resolve");
        require(samples[0].resolved && std::abs(samples[0].gap+.0001)<1e-11,"incorrect signed STEP probe gap");
        require(std::abs(force.value[1]-.3)<1e-8 &&
                std::abs(force.value[0]+.075/std::hypot(1.,.001))<1e-8,"incorrect virtual work or friction");
        pair.material.maximum_depth=.00005;
        require(!add_contact_forces(nullptr,bodies,pairs,velocity,work,&samples),"overlap limit did not reject");
        bodies[1].pose.origin[0]=.002;
        require(add_contact_forces(nullptr,bodies,pairs,velocity,work,&samples) && !samples[0].resolved,
                "finite face had a false contact outside its trim");
    }
    const auto plate=BRepPrimAPI_MakeBox(gp_Pnt(-5,-5,-1),10,10,1).Shape();
    const auto hole=BRepPrimAPI_MakeBox(gp_Pnt(-1,-1,-2),2,2,3).Shape();
    const auto perforated=contact_surface(BRepAlgoAPI_Cut(plate,hole).Shape());
    ContactWork work;
    require(!perforated->first_hit({0,0,.002},{0,0,-1},.004,work),
            "compiled planar trim filled a rectangular hole");
    const auto rim=perforated->first_hit({.002,0,.002},{0,0,-1},.004,work);
    require(rim && std::abs(rim->distance-.002)<1e-11,"compiled planar trim missed the supported area");
}

void check_pair_contact(const TopoDS_Shape& a, const TopoDS_Shape& b) {
    using namespace nc::dynamics;
    const std::vector<ContactBodyState> bodies{{nullptr,{},{{}}}, {nullptr,{},{{{0,0,1},{}}}}};
    for (const auto backend : {PairBackend::occt, PairBackend::mesh, PairBackend::brep}) {
        const auto first=pair_geometry(a,backend,5e-6),second=pair_geometry(b,backend,5e-6);
        ContactPair pair{0,1,{},{},ContactDirection::world,{},{1000,1,0,0,.001,.001},.0006,
                         pair_contact_query(first,second)};
        Forces force{{0},{0},{0}};
        ContactWork work;
        std::vector<ContactSample> observations;
        require(add_contact_forces(&force,bodies,{&pair,1},{0},work,&observations) &&
                observations[0].resolved && std::abs(force.value[0]-.1)<1e-9,
                "closest-pair gap or virtual work disagrees with the analytic result");
        if (backend==PairBackend::brep) {
            const auto event=fixed_axis_pair_sweep(*first,*second,{},{},{},{{0,0,-.0005}},.0001,1e-4);
            require(event.status==SweepStatus::hit && event.safe_prefix<=.8 && event.witness_time.value_or(0)>=.8,
                    "closest-pair timestep CCD misses the reference event");
            pair.closest=pair_contact_query(first,second,{1e-9,1});
            require(!add_contact_forces(nullptr,bodies,{&pair,1},{0},work) && work.unresolved_queries==1,
                    "B-rep budget exhaustion was mistaken for separation");
        }
    }
}

void check_contained_contact() {
    using namespace nc::dynamics;
    const auto blank=BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0,0,-1),gp_Dir(0,0,1)),2.,2.).Shape();
    const auto tool=BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0,0,-2),gp_Dir(0,0,1)),1.,4.).Shape();
    const auto bore=BRepAlgoAPI_Cut(blank,tool).Shape();
    const auto pin=BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0,0,-1),gp_Dir(0,0,1)),.8,2.).Shape();
    const ContactPair pair{0,1,{},{},ContactDirection::contained,{-1,0,0},{1000,1,2,0,.001,.0003}};
    for (double mesh:{0.,.000001}) {
        const auto a=contact_surface(bore,mesh),b=contact_surface(pin,mesh);
        std::vector<ContactBodyState> bodies{{a.get(),{},{{}}}, {b.get(),{},{{{1,0,0},{}}}}};
        ContactWork work;
        std::vector<ContactSample> samples;
        const std::span pairs(&pair,1);
        const Vector velocity{0};
        require(add_contact_forces(nullptr,bodies,pairs,velocity,work,&samples) &&
                samples[0].resolved && std::abs(samples[0].gap-.0002)<2e-6,
                "concentric bore clearance did not resolve");
        bodies[1].pose.origin[0]=.00025;
        bodies[1].prescribed.linear[0]=.1;
        Forces force{Vector(1),Vector(1),Vector(1)};
        require(add_contact_forces(&force,bodies,pairs,velocity,work,&samples) &&
                std::abs(samples[0].gap+.00005)<2e-6 && std::abs(force.value[0]+.25)<.003,
                "contained signed depth or prescribed force is wrong");
        bodies[1].pose.origin[0]=.0014;
        require(!add_contact_forces(nullptr,bodies,pairs,velocity,work,&samples),
                "a deeply escaped pin switched to the far bore wall");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        check_contained_contact();
        const fs::path fixtures = argc == 2 ? fs::path(argv[1]) : fs::path("step_fixtures");
        fs::create_directories(fixtures);
        const auto block_a = BRepPrimAPI_MakeBox(gp_Pnt(-0.1, -0.1, -0.2), 0.2, 0.2, 0.2).Shape();
        const auto block_b = BRepPrimAPI_MakeBox(gp_Pnt(-0.1, -0.1, 0.5), 0.2, 0.2, 0.2).Shape();
        const auto plate = BRepPrimAPI_MakeBox(gp_Pnt(-1.5, -1.5, -0.1), 3, 3, 0.2).Shape();
        const auto tool = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -0.3), gp_Dir(0, 0, 1)), 1, 0.6).Shape();
        const auto bore = BRepAlgoAPI_Cut(plate, tool).Shape();
        const auto pin = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -0.6), gp_Dir(0, 0, 1)), 0.25, 0.1).Shape();
        const auto guard = BRepPrimAPI_MakeSphere(gp_Pnt(0.9, 0.9, 0), 0.002).Shape();
        const auto rotor = BRepPrimAPI_MakeSphere(gp_Pnt(1, 0, 0), 0.002).Shape();
        write_step(block_a, fixtures / "block-a.step");
        write_step(block_b, fixtures / "block-b.step");
        write_step(bore, fixtures / "bore.step");
        write_step(pin, fixtures / "pin.step");
        write_step(guard, fixtures / "guard.step");
        write_step(rotor, fixtures / "rotor.step");

        TColgp_Array2OfPnt points(1, 4, 1, 3);
        TColStd_Array2OfReal weights(1, 4, 1, 3);
        for (int i = 1; i <= 4; ++i) {
            for (int j = 1; j <= 3; ++j) {
                points(i, j) = gp_Pnt(i * 0.1, j * 0.1, 0.02 * (i - 2) * (j - 2));
                weights(i, j) = 1.0 + 0.05 * (i + j);
            }
        }
        TColStd_Array1OfReal u_knots(1, 3), v_knots(1, 2);
        TColStd_Array1OfInteger u_counts(1, 3), v_counts(1, 2);
        u_knots(1) = v_knots(1) = 0;
        u_knots(2) = 0.5;
        u_knots(3) = v_knots(2) = 1;
        u_counts(1) = u_counts(3) = v_counts(1) = v_counts(2) = 3;
        u_counts(2) = 1;
        const Handle(Geom_BSplineSurface) freeform =
            new Geom_BSplineSurface(points, weights, u_knots, v_knots, u_counts, v_counts, 2, 2);
        const auto spline_face = BRepBuilderAPI_MakeFace(freeform, 0, 1, 0, 1, 1e-7).Shape();
        write_step(spline_face, fixtures / "freeform.step");
        const auto spline = read_step(fixtures / "freeform.step");
        require(spline.patches.size() == 2, "rational B-spline did not split at the interior knot");
        for (const auto& patch : spline.patches) {
            const double u = patch.domain.u.center();
            const double v = patch.domain.v.center();
            const auto actual = midpoint(evaluate(patch.surface, {u, v}));
            const auto expected = freeform->Value(u, v);
            require(std::abs(actual[0] - expected.X()) < 1e-10 && std::abs(actual[1] - expected.Y()) < 1e-10 &&
                        std::abs(actual[2] - expected.Z()) < 1e-10, "Bezier extraction changed spline evaluation");
        }
        const auto freeform_contact=contact_surface(read_step_shape(fixtures/"freeform.step"));
        const auto exact_point=freeform->Value(.37,.42);
        dynamics::ContactWork freeform_work;
        const auto freeform_ray=freeform_contact->first_hit(
            {.001*exact_point.X(),.001*exact_point.Y(),.0001},{0,0,-1},.0002,freeform_work);
        require(freeform_ray && std::abs(freeform_ray->distance-(.0001-.001*exact_point.Z()))<1e-9,
                "shared directional solver missed the rational STEP patch");
        const auto shape_a = read_step_shape(fixtures / "block-a.step");
        const auto shape_b = read_step_shape(fixtures / "block-b.step");
        check_directional_contact(shape_a,shape_b);
        check_pair_contact(shape_a,shape_b);
        BRepExtrema_DistShapeShape reference{shape_a, shape_b};
        require(reference.IsDone(), "OCCT reference distance failed");
        const PreparedModel a{adapt_occt(shape_a)};
        const PreparedModel b{adapt_occt(shape_b)};
        Options options;
        options.distance_tolerance = 1e-5;
        const auto gap = distance(a, b, {}, 0, options);
        require(gap.lower <= 0.5 && 0.5 <= gap.upper, "imported box separation not enclosed");
        require(gap.upper - gap.lower <= options.distance_tolerance, "imported box accuracy missed");
        require(std::abs(reference.Value() - 0.5) < 1e-10, "OCCT reference disagrees with analytic distance");
        const Motion downward{{}, {0, 0, -1}, {0, 0, 1}, 0, 0};
        const auto box_hit = sweep(a, b, downward, {0, 1}, 0.1, options);
        require(box_hit.status == SweepStatus::hit && box_hit.safe_prefix <= 0.4 &&
                    box_hit.witness_time.value_or(0) >= 0.4,
                "imported box sweep misses reference event");
        const PreparedModel bore_model{read_step(fixtures / "bore.step")};
        const PreparedModel pin_model{read_step(fixtures / "pin.step")};
        const Motion through{{}, {0, 0, 1.2}, {0, 0, 1}, 0, 0};
        const auto clear = sweep(bore_model, pin_model, through, {0, 1}, 0.09, options);
        require(clear.status == SweepStatus::clear, "pin passage through circular hole not certified");
        options.distance_tolerance = 2e-4;
        options.time_tolerance = 5e-4;
        const double rim_time = (0.4 - std::sqrt(0.09 * 0.09 - 0.07 * 0.07)) / 1.2;
        for (double clock_offset : {0.0, -1.0}) {
            const Motion eccentric{{0.68, 0, -1.2 * clock_offset}, {0, 0, 1.2}, {0, 0, 1}, 0, 0};
            const auto rim = sweep(bore_model, pin_model, eccentric, {clock_offset, clock_offset + 1}, 0.09, options);
            require(rim.status == SweepStatus::hit && rim.witness_time && rim.witness,
                    "eccentric pin sweep has no feasible rim witness");
            require(rim.safe_prefix <= rim_time + clock_offset &&
                        *rim.witness_time >= rim_time + clock_offset &&
                        *rim.witness_time - rim.safe_prefix <= options.time_tolerance,
                    "eccentric pin sweep failed the analytic bracket");
            require(rim.statistics.axial_clipped > 0 && rim.statistics.visited < 1'000,
                    "eccentric pin sweep lost the axial interval reduction");
        }
        std::cout << "STEP round-trip cases passed\n";
    } catch (const Standard_Failure& e) {
        std::cerr << e.GetMessageString() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
