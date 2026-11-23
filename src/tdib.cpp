#include "nurbscontact/tdib.hpp"

#include <memory>
#include <queue>

namespace nc {
namespace {
struct Endpoints { IVec first, last; };
struct Projection { Interval first, last; };

HPoint average(HPoint a,const HPoint& b) {
    for (int k=0;k<4;++k) {
        if (a[k].lo==b[k].lo && a[k].hi==b[k].hi) continue;
        const auto mean=a[k]*.5+b[k]*.5;
        a[k]={std::max(std::min(a[k].lo,b[k].lo),mean.lo),std::min(std::max(a[k].hi,b[k].hi),mean.hi)};
    }
    return a;
}

std::array<std::vector<HPoint>,4> triangle_quarters(const std::vector<HPoint>& p) {
    // Blossom evaluated at the three original vertices and the three edge midpoints.
    constexpr int index[3][3]={{0,1,3},{1,2,4},{3,4,5}};
    constexpr int vertex[6][2]={{0,0},{1,1},{2,2},{0,1},{1,2},{2,0}};
    HPoint blossom[6][6];
    for (int i=0;i<6;++i) {
        HPoint row[3];
        for (int k=0;k<3;++k) row[k]=average(p[index[vertex[i][0]][k]],p[index[vertex[i][1]][k]]);
        for (int j=0;j<=i;++j) blossom[i][j]=blossom[j][i]=average(row[vertex[j][0]],row[vertex[j][1]]);
    }
    constexpr int corners[4][3]={{0,3,5},{3,1,4},{5,4,2},{4,5,3}};
    std::array<std::vector<HPoint>,4> halves;
    for (int k=0;k<4;++k) {
        const auto [w,u,v]=corners[k];
        halves[k]={blossom[w][w],blossom[w][u],blossom[u][u],blossom[w][v],blossom[u][v],blossom[v][v]};
    }
    return halves;
}

// Restrict a homogeneous Bézier net to both sides of its midpoint. Equal
// endpoints (including constant weights) stay exact under a convex blend.
std::array<std::vector<HPoint>,2> bisect(const std::vector<HPoint>& net,std::size_t nu,std::size_t nv,bool u) {
    std::array<std::vector<HPoint>,2> half{net,net};
    const auto size=u ? nu : nv, stride=u ? nv : 1, curves=u ? nv : nu;
    std::vector<HPoint> work(size);
    for (std::size_t c=0;c<curves;++c) {
        const auto start=u ? c : c*nv;
        for (std::size_t i=0;i<size;++i) work[i]=net[start+i*stride];
        for (std::size_t level=1;level<size;++level) {
            for (std::size_t i=0;i<size-level;++i) work[i]=average(work[i],work[i+1]);
            half[0][start+stride*level]=work.front();
            half[1][start+stride*(size-1-level)]=work[size-1-level];
        }
    }
    return half;
}

struct Patch {
    MovingBezier net;
    std::vector<Endpoints> points;
    std::array<std::unique_ptr<Patch>,4> children;
    Rect domain;
    double width() const { return std::max(domain.u.width(),domain.v.width()); }

    Patch(MovingBezier data,Rect r) : net(std::move(data)),domain(r) {
        for (std::size_t i=0;i<net.first.size();++i)
            points.push_back({cartesian(net.first[i]),cartesian(net.last[i])});
    }

    void subdivide() {
        if (children[0]) return;
        if (net.layout==BezierLayout::quadratic_triangle) {
            auto first=triangle_quarters(net.first), last=triangle_quarters(net.last);
            const Rect local{{0,width()*.5},{0,width()*.5}};
            for (int i=0;i<4;++i) children[i]=std::make_unique<Patch>(
                MovingBezier{3,3,std::move(first[i]),std::move(last[i]),net.layout},local);
            return;
        }
        const bool u=domain.u.width()>=domain.v.width();
        auto first=bisect(net.first,net.nu,net.nv,u),last=bisect(net.last,net.nu,net.nv,u);
        const auto [left,right]=split(domain,u);
        for (int i=0;i<2;++i) children[i]=std::make_unique<Patch>(
            MovingBezier{net.nu,net.nv,std::move(first[i]),std::move(last[i])},i==0 ? left : right);
    }

    Vec direction(bool u,double t) const {
        const auto edge=[&](std::size_t p,std::size_t q) {
            Vec delta{};
            for (int k=0;k<3;++k)
                delta[k]=std::lerp(points[q].first[k].center()-points[p].first[k].center(),
                                   points[q].last[k].center()-points[p].last[k].center(),t);
            return delta;
        };
        const auto last=points.size()-1;
        if (net.layout==BezierLayout::quadratic_triangle) return edge(0,u ? 2 : 5);
        const auto offset=u ? last+1-net.nv : net.nv-1;
        auto d=edge(0,offset);
        const auto opposite=edge(last-offset,last);
        for (int k=0;k<3;++k) d[k]=.5*d[k]+.5*opposite[k];
        return d;
    }
};

// For t in [0,1], interpolation of outward-rounded endpoint ordinates
// bounds the entire line, including its coefficient-construction error.
// max(A) >= min(B) iff at least one pair of lines satisfies A_i >= B_j.
// Enumerating pairs avoids branch-sensitive sorting of nearly parallel lines.
// Keep a hull of feasible intervals; an interior gap may be revisited later.
bool clip(const std::vector<Projection>& a,const std::vector<Projection>& b,double padding,Interval& time) {
    Interval feasible{1,0};
    for (const auto& p:a) for (const auto& q:b) {
        const double f0=up(up(p.first.hi-q.first.lo)+padding);
        const double f1=up(up(p.last.hi-q.last.lo)+padding);
        if (!std::isfinite(f0) || !std::isfinite(f1)) return true;
        if (f0<0 && f1<0) continue;
        Interval part=time;
        if (f0<0) part.lo=std::max(part.lo,down(-f0/up(f1-f0)));
        else if (f1<0) {
            const double denominator=down(f0-f1);
            if (denominator>0) part.hi=std::min(part.hi,up(f0/denominator));
        }
        if (part.lo>part.hi) continue;
        feasible=hull(feasible,part);
        if (feasible.lo==time.lo && feasible.hi==time.hi) return true;
    }
    if (feasible.lo>feasible.hi) return false;
    time=feasible;
    return true;
}

Vec cross_axis(Vec a,Vec b) {
    return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}

struct Cell {
    Patch *a,*b;
    Interval time;
    bool clipped{};
    bool operator<(const Cell& other) const {
        if (time.lo!=other.time.lo) return time.lo>other.time.lo;
        return a->width()+b->width()>other.a->width()+other.b->width();
    }
};

bool overlaps(Cell& cell,double margin) {
    std::vector<Projection> a(cell.a->points.size()),b(cell.b->points.size());
    const auto project=[&](Vec n,int coordinate) {
        double scale=std::max({std::abs(n[0]),std::abs(n[1]),std::abs(n[2])});
        if (!(scale>0 && std::isfinite(scale))) return true;
        for (auto& v:n) v/=scale;
        const auto fill=[&](const Patch& patch,std::vector<Projection>& lines) {
            for (std::size_t i=0;i<lines.size();++i) {
                const auto& p=patch.points[i];
                if (coordinate>=0) lines[i]={p.first[coordinate],p.last[coordinate]};
                else lines[i]={dot(p.first,exact(n)),dot(p.last,exact(n))};
            }
        };
        fill(*cell.a,a); fill(*cell.b,b);
        const double padding=margin==0 ? 0. : up(margin*up(up(std::abs(n[0])+std::abs(n[1]))+std::abs(n[2])));
        return clip(a,b,padding,cell.time) && clip(b,a,padding,cell.time);
    };
    for (int k=0;k<3;++k) {
        Vec axis{}; axis[k]=1.;
        if (!project(axis,k)) return false;
    }
    const double t=cell.time.center();
    const auto au=cell.a->direction(true,t),av=cell.a->direction(false,t);
    const auto bu=cell.b->direction(true,t),bv=cell.b->direction(false,t);
    for (Vec axis:{cross_axis(au,av),cross_axis(bu,bv),cross_axis(au,bu),
                   cross_axis(au,bv),cross_axis(av,bu),cross_axis(av,bv)})
        if (!project(axis,-1)) return false;
    return true;
}

void validate(const MovingBezier& p) {
    const bool valid_dimensions=p.layout==BezierLayout::quadratic_triangle ? p.first.size()==6 :
                               p.nu && p.nv && p.nu<=p.first.size()/p.nv && p.nu*p.nv==p.first.size();
    if (!valid_dimensions || p.first.size()!=p.last.size())
        throw std::invalid_argument("invalid TDIB control net dimensions");
    for (std::size_t i=0;i<p.first.size();++i) {
        for (int k=0;k<4;++k) for (auto x:{p.first[i][k],p.last[i][k]})
            if (!std::isfinite(x.lo) || !std::isfinite(x.hi) || x.lo>x.hi)
                throw std::invalid_argument("invalid TDIB control coefficient");
        const auto a=p.first[i][3],b=p.last[i][3];
        if (a.lo<=0 || a.lo!=b.lo || a.hi!=b.hi)
            throw std::invalid_argument("TDIB requires constant positive weights");
    }
}
}

TdibResult tdib_envelopes(const MovingBezier& first,const MovingBezier& second,
                          double width,double horizon,double margin,std::size_t limit) {
    if (!(width>0 && width<=1 && horizon>=0 && horizon<=1 && margin>=0 && std::isfinite(margin)))
        throw std::invalid_argument("invalid TDIB search settings");
    validate(first); validate(second);
    const Rect domain{{0,1},{0,1}};
    Patch a(first,domain),b(second,domain);
    std::priority_queue<Cell> queue;
    queue.push({&a,&b,{0,horizon}});
    std::size_t nodes=0;
    while (!queue.empty()) {
        auto cell=queue.top(); queue.pop();
        if (!cell.clipped) {
            if (nodes>=limit) return {TdibStatus::unresolved,{cell.time.lo,horizon},nodes};
            ++nodes;
            if (!overlaps(cell,margin)) continue;
            // A newly clipped cell may now follow another pending lower bound.
            if (!queue.empty() && cell.time.lo>queue.top().time.lo) {
                cell.clipped=true;
                queue.push(cell);
                continue;
            }
        }
        auto* patch=cell.a->width()>=cell.b->width() ? cell.a : cell.b;
        if (patch->width()<=width && cell.time.width()<=width)
            return {TdibStatus::candidate,cell.time,nodes};
        if (patch->width()<=width) {
            const auto mid=cell.time.center();
            if (mid<=cell.time.lo || mid>=cell.time.hi) return {TdibStatus::unresolved,cell.time,nodes};
            queue.push({cell.a,cell.b,{cell.time.lo,mid}});
            queue.push({cell.a,cell.b,{mid,cell.time.hi}});
        } else {
            patch->subdivide();
            for (const auto& part:patch->children) if (part)
                queue.push({patch==cell.a ? part.get() : cell.a,patch==cell.b ? part.get() : cell.b,cell.time});
        }
    }
    return {TdibStatus::clear,{},nodes};
}
}
