#include "nurbscontact/compiled_part.hpp"
#include "nurbscontact/occt.hpp"
#include "../src/cad_compiled_chart.hpp"
#include <BRepBndLib.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <iostream>

int main(int argc, char **argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("nc_compile_part SOLID.step OUTPUT.ncpart");
        const auto shape = nc::read_step_shape(argv[1]);
        TopTools_IndexedMapOfShape faces, solids;
        TopExp::MapShapes(shape, TopAbs_FACE, faces);
        TopExp::MapShapes(shape, TopAbs_SOLID, solids);
        if (solids.Extent() != 1)
            throw std::runtime_error("expected one solid per CAD body");
        TopTools_IndexedMapOfShape owned_faces;
        TopExp::MapShapes(solids(1), TopAbs_FACE, owned_faces);
        if (owned_faces.Extent() != faces.Extent())
            throw std::runtime_error("input includes faces outside the solid");
        for (TopExp_Explorer shells(solids(1), TopAbs_SHELL); shells.More(); shells.Next())
            if (!BRep_Tool::IsClosed(shells.Current()))
                throw std::runtime_error("open shell in the supplied solid");
        nc::compiled::PartFile data;
        data.geometry = nc::adapt_occt(shape);
        for (int i = 1; i <= faces.Extent(); ++i) {
            auto f = TopoDS::Face(faces(i));
            nc::cad::CompiledChart chart(f);
            Bnd_Box box;
            BRepBndLib::AddOptimal(f, box, false, true);
            double x, y, z, X, Y, Z;
            box.Get(x, y, z, X, Y, Z);
            data.faces.push_back({{{chart.u0, chart.u1}, {chart.v0, chart.v1}},
                                  {x, y, z},
                                  {X, Y, Z},
                                  BRep_Tool::Tolerance(f),
                                  f.Orientation() == TopAbs_REVERSED ? -1 : 1,
                                  chart.rectangular});
        }
        nc::compiled::save(data, argv[2]);
        std::cout << "faces " << data.faces.size() << " patches " << data.geometry.patches.size() << " trims "
                  << data.geometry.trims.size() << "\n";
    } catch (const Standard_Failure &e) {
        std::cerr << e.GetMessageString() << '\n';
        return 1;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
