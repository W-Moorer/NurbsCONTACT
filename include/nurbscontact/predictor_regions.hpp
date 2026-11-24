#pragma once

#include "nurbscontact/contact_dynamics.hpp"

namespace nc::dynamics {

// Transport loaded material points within a Newton solve. An empty predictor
// or an opening support must be queried at the corrected trial pose.
class PredictorRegions final : public PairRegionQuery {
    struct Anchors {Vec a,b,normal_a,normal_b;};
    std::shared_ptr<const PairRegionQuery> exact_;
    mutable std::vector<Anchors> anchors_;
    mutable bool ready_{};
    mutable std::optional<std::array<ContactPose,2>> last_pose_;
    mutable std::vector<ContactRegion> exact_points_;
    const std::vector<ContactRegion>& query(ContactPose a,ContactPose b,ContactWork& work) const {
        const auto same=[](const ContactPose& x,const ContactPose& y) {
            return x.origin==y.origin && x.axis==y.axis && x.angle==y.angle;
        };
        if (!last_pose_ || !same(a,(*last_pose_)[0]) || !same(b,(*last_pose_)[1])) {
            exact_points_=exact_->regions(a,b,work);
            last_pose_=std::array{a,b};
        }
        return exact_points_;
    }
public:
    explicit PredictorRegions(std::shared_ptr<const PairRegionQuery> exact):exact_(std::move(exact)) {}
    void reset() {ready_=false;anchors_.clear();}
    std::size_t independent_mesh_faces() const override {return ready_ ? 0 : exact_->independent_mesh_faces();}
    std::vector<ContactRegion> regions(ContactPose a,ContactPose b,ContactWork& work) const override {
        if (!ready_) {
            const auto& points=query(a,b,work);
            anchors_.clear();
            for (const auto& p:points) {
                const auto half=scale(p.normal,.5*p.depth_m);
                anchors_.push_back({a.inverse_rotate(sub(add(p.point,half),a.origin)),
                    b.inverse_rotate(sub(sub(p.point,half),b.origin)),
                    a.inverse_rotate(p.normal),b.inverse_rotate(p.normal)});
            }
            ready_=true;
            return points;
        }
        std::vector<ContactRegion> points;
        points.reserve(anchors_.size());
        for (const auto& p:anchors_) {
            const auto pa=a.point(p.a),pb=b.point(p.b);
            Vec n=add(a.rotate(p.normal_a),b.rotate(p.normal_b));
            const double norm=length(n);
            if (norm<1.) throw UnresolvedContact("predictor contact frame rotated out of range");
            n=scale(n,1/norm);
            const double depth=dot3(sub(pa,pb),n);
            if (depth>0) points.push_back({scale(add(pa,pb),.5),n,depth});
        }
        // Query fresh without replacing the predictor's anchors. A rejected
        // line-search trial must not change the force law for the next trial.
        if (points.empty() || points.size()!=anchors_.size()) return query(a,b,work);
        return points;
    }
    SurfaceTransition transition(ContactPose a,ContactPose b,ContactPose c,ContactPose d,double eps) const override {
        return exact_->transition(a,b,c,d,eps);
    }
    SurfaceTransition transition(ContactPose a,ContactPose b,ContactPose c,ContactPose d,double eps,double time_eps) const override {
        return exact_->transition(a,b,c,d,eps,time_eps);
    }
};

}  // namespace nc::dynamics
