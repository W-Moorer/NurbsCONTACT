#include "cpu_mesh_window.hpp"
#include "contact_tasks.hpp"
#include <algorithm>
#include <numeric>

namespace nc::cpu {
namespace {
// Axis intervals are clipped to the contact slab. The diagonal intervals
// bound the full original triangle; a point at distance pad changes any
// diagonal projection by at most sqrt(2)*pad.
struct Bounds {
    std::array<double,9> lo,hi;
    bool overlaps(const Bounds& b) const {
        for (std::size_t k=0;k<lo.size();++k)
            if (lo[k]>b.hi[k] || b.lo[k]>hi[k]) return false;
        return true;
    }
};
Bounds bounds(const WindowTriangle& c,double pad) {
    Bounds box;
    std::copy(c.lo.begin(),c.lo.end(),box.lo.begin());
    std::copy(c.hi.begin(),c.hi.end(),box.hi.begin());
    std::size_t k=3;
    for (int i=0;i<3;++i) {
        const int j=(i+1)%3;
        for (int sign:{1,-1}) {
            const auto project=[&](const Vec& p) {return p[i]+sign*p[j];};
            const double a=project(c.triangle[0]), b=project(c.triangle[1]), d=project(c.triangle[2]);
            box.lo[k]=std::min({a,b,d})-2*pad;
            box.hi[k]=std::max({a,b,d})+2*pad;
            ++k;
        }
    }
    return box;
}
struct Leaf { Bounds box; std::size_t face; };
struct Node {
    Bounds box;
    std::size_t first,end,left,right;
};
class WindowTree {
    std::vector<Leaf>& leaves;
    std::vector<Node> nodes;
    std::size_t build(std::size_t first,std::size_t end) {
        const auto index=nodes.size();
        auto box=leaves[first].box;
        for (auto i=first+1;i<end;++i) for (std::size_t k=0;k<box.lo.size();++k) {
            box.lo[k]=std::min(box.lo[k],leaves[i].box.lo[k]);
            box.hi[k]=std::max(box.hi[k],leaves[i].box.hi[k]);
        }
        nodes.push_back({box,first,end,0,0});
        if (end-first>8) {
            int axis=0;
            for (int k=1;k<3;++k) if (box.hi[k]-box.lo[k]>box.hi[axis]-box.lo[axis]) axis=k;
            const auto mid=first+(end-first)/2;
            std::nth_element(leaves.begin()+first,leaves.begin()+mid,leaves.begin()+end,
                [&](const Leaf& a,const Leaf& b) {return a.box.lo[axis]+a.box.hi[axis]<b.box.lo[axis]+b.box.hi[axis];});
            const auto left=build(first,mid);
            const auto right=build(mid,end);
            nodes[index].left=left;
            nodes[index].right=right;
        }
        return index;
    }
public:
    explicit WindowTree(std::vector<Leaf>& items):leaves(items) {
        nodes.reserve(items.size());
        build(0,items.size());
    }
    void query(std::size_t i,const Leaf& source,std::vector<std::array<std::size_t,2>>& out,
               const TrianglePairFilter& filter,std::size_t& tests) const {
        const auto& node=nodes[i];
        if (!node.box.overlaps(source.box)) return;
        if (node.left) {
            query(node.left,source,out,filter,tests);
            query(node.right,source,out,filter,tests);
        } else {
            for (auto j=node.first;j<node.end;++j) {
                if (!source.box.overlaps(leaves[j].box)) continue;
                ++tests;
                if (!filter.keep || filter.keep(source.face,leaves[j].face))
                    out.push_back({source.face,leaves[j].face});
            }
        }
    }
};
}
std::size_t window_pairs(std::span<const WindowTriangle> candidates,double pad,
                         std::vector<std::array<std::size_t,2>>& pairs,const TrianglePairFilter& filter) {
    std::array<std::vector<Leaf>,2> sides;
    for (const auto& c:candidates) {
        sides[c.side].push_back({bounds(c,pad),c.face});
        if (c.side && filter.place_target) filter.place_target(c.face);
    }
    if (sides[0].empty() || sides[1].empty()) return 0;
    const WindowTree target(sides[1]);
    constexpr std::size_t grain=32;
    std::vector<std::vector<std::array<std::size_t,2>>> chunks((sides[0].size()+grain-1)/grain);
    std::vector<std::size_t> tested(chunks.size());
    contact_tasks::ranges(chunks.size(),1,[&](auto first,auto end) {
        for (auto k=first;k<end;++k) {
            std::size_t tests=0;
            for (auto i=k*grain;i<std::min((k+1)*grain,sides[0].size());++i)
                target.query(0,sides[0][i],chunks[k],filter,tests);
            tested[k]=tests;
        }
    });
    std::size_t count=pairs.size();
    for (const auto& chunk:chunks) count+=chunk.size();
    pairs.reserve(count);
    for (const auto& chunk:chunks) pairs.insert(pairs.end(),chunk.begin(),chunk.end());
    return std::accumulate(tested.begin(),tested.end(),std::size_t{});
}
}
