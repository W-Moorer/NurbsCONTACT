#pragma once
#include "nurbscontact/cad_contact_regions.hpp"

namespace nc::cad {
// Prepare the standard CPU search grid and all four pressure integration rules.
void prepare_patch_part(const TopoDS_Shape&,bool bezier=false);
struct PatchDiscovery {
    std::vector<std::optional<CadChord>> witnesses;
    std::vector<int> labels;
    std::size_t candidate_pairs,attempts,tracked,tests;
    double query_s,broad_s,group_s;
    int window_builds{},window_reuses{},reused_paths{};
};
enum class PressureFailure { none, no_columns, no_resultant, centroid_chord };
class PatchSearch {
public:
    PatchSearch(const TopoDS_Shape&,const TopoDS_Shape&,int grid,double reach,bool bezier=false);
    ~PatchSearch();
    void prepare_dense() const;
    std::array<std::size_t,2> asset_counts() const;
    std::size_t skipped_nodes() const;
    std::array<std::size_t,4> ray_counts() const;
    bool boxes_separate(const std::array<dynamics::ContactPose,2>& poses) const;
    PatchDiscovery discover(const std::array<dynamics::ContactPose,2>& poses,bool indexed,bool group);
    std::optional<CadChord> chord(const std::array<dynamics::ContactPose,2>& poses,Vec p,Vec n,
                                  std::array<std::size_t,2> ids);
    std::optional<dynamics::ContactRegion> integrate(const std::array<dynamics::ContactPose,2>& poses,
                                                   const std::vector<CadChord>& group,int order,int worker=0,
                                                   PressureFailure* failure=nullptr);
private:
    struct Data;
    std::unique_ptr<Data> data_;
};
}
