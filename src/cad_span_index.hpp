#pragma once
#include <gp_XYZ.hxx>
#include <algorithm>
#include <vector>

namespace nc::cad {
// Contiguous parameter ranges preserve the original order of ray root tests.
// The box at a branch contains all Euclidean control hulls below it.
class SpanIndex {
    struct Node {
        gp_XYZ lo,hi;
        std::size_t begin,end,right{};
    };
    std::vector<Node> nodes_;
    template<class Visit>
    bool visit(std::size_t id,const gp_XYZ& lo,const gp_XYZ& hi,Visit& test) const {
        const auto& n=nodes_[id];
        if (hi.X()<n.lo.X() || lo.X()>n.hi.X() || hi.Y()<n.lo.Y() || lo.Y()>n.hi.Y() ||
            hi.Z()<n.lo.Z() || lo.Z()>n.hi.Z()) return true;
        if (n.right) return visit(id+1,lo,hi,test) && visit(n.right,lo,hi,test);
        for (auto i=n.begin;i<n.end;++i) if (!test(i)) return false;
        return true;
    }
public:
    template<class BoxAt> void build(std::size_t count,BoxAt box) {
        nodes_.clear();
        if (!count) return;
        const auto insert=[&](auto&& self,std::size_t a,std::size_t b)->std::size_t {
            const auto id=nodes_.size();
            nodes_.push_back({box(a).lo,box(a).hi,a,b});
            if (b-a<=8) {
                auto& n=nodes_[id];
                for (auto i=a+1;i<b;++i) for (int d=1;d<=3;++d) {
                    n.lo.SetCoord(d,std::min(n.lo.Coord(d),box(i).lo.Coord(d)));
                    n.hi.SetCoord(d,std::max(n.hi.Coord(d),box(i).hi.Coord(d)));
                }
            } else {
                const auto mid=a+(b-a)/2;
                const auto left=self(self,a,mid),right=self(self,mid,b);
                auto& n=nodes_[id];
                n.right=right;
                for (int d=1;d<=3;++d) {
                    n.lo.SetCoord(d,std::min(nodes_[left].lo.Coord(d),nodes_[right].lo.Coord(d)));
                    n.hi.SetCoord(d,std::max(nodes_[left].hi.Coord(d),nodes_[right].hi.Coord(d)));
                }
            }
            return id;
        };
        insert(insert,0,count);
    }
    template<class Visit> bool query(const gp_XYZ& lo,const gp_XYZ& hi,Visit test) const {
        return nodes_.empty() || visit(0,lo,hi,test);
    }
};
}
