#pragma once
#include "nurbscontact/cpu_mesh.hpp"

namespace nc::cpu {
class FiniteFeatures {
public:
    FiniteFeatures(const MeshData& a,const MeshData& b,MeshScene& scene,double max_depth);
    ~FiniteFeatures();
    std::vector<dynamics::ContactRegion> query(const std::array<RigidPlacement,2>& poses,dynamics::ContactWork& work);
private:
    struct Data;
    std::unique_ptr<Data> data_;
};
}
