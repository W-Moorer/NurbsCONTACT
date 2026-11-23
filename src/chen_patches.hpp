#pragma once
#include <Geom_BezierSurface.hxx>
#include <TopoDS_Shape.hxx>
#include <vector>

namespace nc::cad {
// Bounded supporting charts; the original trimmed solids validate proposals.
std::vector<Handle(Geom_BezierSurface)> bezier_charts(const TopoDS_Shape& shape);
}
