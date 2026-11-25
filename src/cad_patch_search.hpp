#pragma once
#include "nurbscontact/cad_contact_regions.hpp"

namespace nc::cad {
struct PatchDiscovery {
    std::vector<std::optional<CadChord>> witnesses;
    std::vector<int> labels;
    std::size_t candidate_pairs,attempts,tracked,tests;
    double query_s,broad_s,group_s;
};
class PatchSearch {
public:
    PatchSearch(const TopoDS_Shape&,const TopoDS_Shape&,int grid,double reach);
    ~PatchSearch();
    bool boxes_separate(const std::array<dynamics::ContactPose,2>& poses) const;
    PatchDiscovery discover(const std::array<dynamics::ContactPose,2>& poses,bool indexed,bool group);
    std::optional<CadChord> chord(const std::array<dynamics::ContactPose,2>& poses,Vec p,Vec n,
                                  std::array<std::size_t,2> ids);
    std::optional<dynamics::ContactRegion> integrate(const std::array<dynamics::ContactPose,2>& poses,
                                                   const std::vector<CadChord>& group,int order,int worker=0);
private:
    struct Data;
    std::unique_ptr<Data> data_;
};
}
