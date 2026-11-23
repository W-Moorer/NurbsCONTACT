#pragma once
#include "nurbscontact/cad_solid_contact.hpp"
#include <gp_Ax1.hxx>
#include <gp_Quaternion.hxx>
#include <gp_Trsf.hxx>
#include <functional>

namespace nc::cad {
inline gp_Pnt point(Vec v) { return {v[0],v[1],v[2]}; }
inline Vec coordinates(const gp_Pnt& p) { return {p.X(),p.Y(),p.Z()}; }
inline gp_Trsf placement(dynamics::ContactPose pose) {
    gp_Trsf t;
    if (pose.angle!=0) t.SetRotation(gp_Ax1({},gp_Dir(pose.axis[0],pose.axis[1],pose.axis[2])),pose.angle);
    t.SetTranslationPart(gp_Vec(1000*pose.origin[0],1000*pose.origin[1],1000*pose.origin[2]));
    return t;
}

// Linear interpolation of the pivot and shortest-arc rotation about it.
struct RigidPath {
    gp_Trsf start,end;
    gp_Pnt pivot,from,to;
    gp_Quaternion q0;
    gp_Vec axis;
    double angle;
    RigidPath(dynamics::ContactPose a,dynamics::ContactPose b,Vec center)
        : start(placement(a)),end(placement(b)),pivot(point(dynamics::scale(center,1000))),
          from(pivot.Transformed(start)),to(pivot.Transformed(end)),q0(start.GetRotation()) {
        auto relative=q0.Inverted()*end.GetRotation();
        if (relative.W()<0) relative=-relative;
        relative.GetVectorAndAngle(axis,angle);
    }
    gp_Trsf at(double t) const {
        if (t==0) return start;
        if (t==1) return end;
        gp_Trsf pose;
        pose.SetRotation(angle==0 ? q0 : q0*gp_Quaternion(axis,angle*t));
        const gp_XYZ p=from.XYZ()*(1-t)+to.XYZ()*t;
        pose.SetTranslationPart(gp_Vec(p-pivot.Transformed(pose).XYZ()));
        return pose;
    }
};
using CandidateQuery=std::function<std::optional<double>(const RigidPath&,const RigidPath&,dynamics::SurfaceTransition&)>;
#ifdef NC_WITH_CHEN
CandidateQuery chen_candidates(const TopoDS_Shape&,const TopoDS_Shape&);
#endif
}
