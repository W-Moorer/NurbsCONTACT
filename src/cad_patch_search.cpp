// Independent CAD-only proposals for the region-discovery replay. A finite
// set of witnesses proves local overlap, but an empty set cannot prove separation.
#include "nurbscontact/cad_contact_regions.hpp"
#include "cad_compiled_chart.hpp"
#include "cad_rigid_path.hpp"
#include "cad_patch_search.hpp"
#include "cad_patch_assets.hpp"
#include "cad_bezier_ray.hpp"
#include <BndLib_AddSurface.hxx>
#include "cad_local_solid.hpp"
#include <BRepAdaptor_Surface.hxx>
#include <BRepTopAdaptor_FClass2d.hxx>
#include <Bnd_Box.hxx>
#include <NCollection_UBTreeFiller.hxx>
#include <OSD_ThreadPool.hxx>
#include <TopExp.hxx>
#include <TopoDS.hxx>
#include <chrono>
#include <deque>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <unordered_map>

namespace {
using namespace nc;
using namespace nc::dynamics;
using Tree=NCollection_UBTree<std::size_t,Bnd_Box>;
using Clock=std::chrono::steady_clock;
struct Chart {
    gp_Pnt point;
    gp_Vec normal;
    Bnd_Box search_box;
    std::size_t face;
    bool interior;
    double u,v;
};
class Selection final : public Tree::Selector {
public:
    explicit Selection(Bnd_Box b): box(b) {}
    bool Reject(const Bnd_Box& branch) const override {return box.IsOut(branch);}
    bool Accept(const std::size_t& id) override {ids.push_back(id); return true;}
    Bnd_Box box;
    std::vector<std::size_t> ids;
};
struct Box {
    Vec center,half;
    Box() = default;
    explicit Box(const Bnd_Box& b) {
        const auto lo=cad::coordinates(b.CornerMin()),hi=cad::coordinates(b.CornerMax());
        center=scale(add(lo,hi),.5); half=scale(sub(hi,lo),.5);
    }
    bool overlaps(const Box& b) const {
        for (int i=0;i<3;++i) if (std::abs(center[i]-b.center[i])>half[i]+b.half[i]) return false;
        return true;
    }
};
// Keep tile and point bounds in contiguous median-split trees. A transformed
// parent bounds every transformed child AABB, including the search margin.
struct ChartTree {
    struct Node {
        Box box;
        int left = -1, right = -1;
        std::size_t chart{}, count{};
    };
    std::vector<Node> nodes;
    void build(std::vector<std::size_t>& ids, const std::vector<Chart>& charts, bool points,
               const std::vector<Bnd_Box>& enclosures = {}) {
        const auto recurse=[&](auto&& self,auto first,auto last)->int {
            Bnd_Box bounds;
            for (auto it=first;it!=last;++it) {
                if (points)
                    bounds.Add(charts[*it].point);
                else
                    bounds.Add(enclosures[*it]);
            }
            int id=static_cast<int>(nodes.size());
            nodes.push_back({Box(bounds)});
            nodes.back().count = static_cast<std::size_t>(last - first);
            if (last-first==1) nodes[id].chart=*first;
            else {
                auto& h=nodes[id].box.half;
                int axis=static_cast<int>(std::max_element(h.begin(),h.end())-h.begin())+1;
                auto middle=first+(last-first)/2;
                std::nth_element(first,middle,last,[&](auto a,auto b) {
                    return charts[a].point.Coord(axis)<charts[b].point.Coord(axis);
                });
                int l=self(self,first,middle),r=self(self,middle,last);
                nodes[id].left=l; nodes[id].right=r;
            }
            return id;
        };
        if (!ids.empty()) recurse(recurse,ids.begin(),ids.end());
    }
};
struct AreaNode {
    Vec point, area;
};
struct NodeBlock {
    std::size_t begin, end;
    gp_XYZ center;
    std::array<gp_XYZ,3> axes;
    cad::CompiledChart::ColumnBox placed(const gp_Trsf& pose) const {
        cad::CompiledChart::ColumnBox box;
        const auto r=pose.VectorialPart();
        box.center=r*center+pose.TranslationPart();
        for (int j=0;j<3;++j) box.axes[j]=r*axes[j];
        for (int d=1;d<=3;++d) {
            double half=4*Precision::Confusion();
            for (const auto& axis:box.axes) half+=std::abs(axis.Coord(d));
            box.lo.SetCoord(d,box.center.Coord(d)-half);
            box.hi.SetCoord(d,box.center.Coord(d)+half);
        }
        return box;
    }
};
struct FaceRule {
    std::vector<AreaNode> nodes;
    std::vector<NodeBlock> blocks;
    struct Column {gp_XYZ point; Vec inward;};
    void index(double reach) {
        std::vector<Column> columns;
        columns.reserve(nodes.size());
        for (const auto& node:nodes) {
            const double weight=length(node.area);
            columns.push_back({cad::point(scale(node.point,1000)).XYZ(),
                               weight>0 ? scale(node.area,-1/weight) : Vec{}});
        }
        for (std::size_t i = 0; i < nodes.size(); i += 8) {
            NodeBlock block{i, std::min(i + 8, nodes.size()), {}, {}};
            auto along=columns[block.end-1].point-columns[i].point;
            if (along.SquareModulus()<1e-24) along=gp_XYZ(1,0,0);
            along.Normalize();
            auto across=along.Crossed(cad::point(columns[i].inward).XYZ());
            if (across.SquareModulus()<1e-12)
                across=along.Crossed(std::abs(along.X())<.8 ? gp_XYZ(1,0,0) : gp_XYZ(0,1,0));
            across.Normalize();
            const std::array axes{along,across,along.Crossed(across)};
            const auto origin=columns[i].point;
            block.center=origin;
            for (int k=0;k<3;++k) {
                double lo=0,hi=0;
                for (auto j=i;j<block.end;++j) {
                    const double start=(columns[j].point-origin).Dot(axes[k]);
                    const double end=start+1000*reach*cad::point(columns[j].inward).XYZ().Dot(axes[k]);
                    lo=std::min({lo,start,end}); hi=std::max({hi,start,end});
                }
                block.center+=axes[k]*(.5*(hi+lo));
                block.axes[k]=axes[k]*(.5*(hi-lo));
            }
            blocks.push_back(block);
        }
    }
};
struct Atlas {
    CadContactIndex index;
    cad::PatchAssets assets;
    std::shared_ptr<const cad::BezierRayAtlas> bezier;
    CadExitQuery rays;
    std::vector<Chart> charts;
    Tree sites,tiles;
    Bnd_Box body_box;
    ChartTree point_index,tile_index;
    std::vector<std::size_t> point_order,tile_order;
    using Face=cad::CompiledChart;
    std::vector<std::unique_ptr<Face>> surfaces;
    std::vector<FaceRule> base_rule;
    struct DenseRules {
        std::array<std::once_flag, 3> once;
        std::array<FaceRule, 3> rules;
    };
    std::deque<DenseRules> dense;
    std::vector<Bnd_Box> face_boxes;
    double column_reach;
    struct StoredChart {
        Vec point, normal, lo, hi, bound_lo, bound_hi;
        std::uint64_t face, interior;
        double u, v;
    };
    Atlas(const TopoDS_Shape& shape, int count, bool use_bezier=false,double reach=0)
        :index(shape),assets(shape),bezier(use_bezier ? std::make_shared<cad::BezierRayAtlas>(shape,assets) : nullptr),
         rays(index,bezier),column_reach(reach) {
        TopTools_IndexedMapOfShape faces;
        TopExp::MapShapes(shape,TopAbs_FACE,faces);
        NCollection_UBTreeFiller<std::size_t,Bnd_Box> points(sites),boxes(tiles);
        const auto face_bounds=index.face_bounds();
        for (int f=1;f<=faces.Extent();++f) {
            Bnd_Box bounds;
            for (const auto xyz : face_bounds[f - 1])
                bounds.Add(cad::point(scale(xyz, 1000)));
            face_boxes.push_back(bounds);
            body_box.Add(bounds);
            const auto face=TopoDS::Face(faces(f));
            surfaces.emplace_back(std::make_unique<Face>(face,true));
        }
        const auto stored = assets.table<StoredChart>(
            "charts-" + std::to_string(count), std::size_t(faces.Extent()) * count * count, [&] {
                std::vector<StoredChart> result;
                for (int f = 1; f <= faces.Extent(); ++f) {
                    const auto face = TopoDS::Face(faces(f));
                    const auto& surface = surfaces[f - 1]->surface;
                    const auto& trims = surfaces[f - 1]->trims;
                    const double u0 = surface.FirstUParameter(), v0 = surface.FirstVParameter();
                    const double du = (surface.LastUParameter() - u0) / count,
                                 dv = (surface.LastVParameter() - v0) / count;
                    for (int i = 0; i < count; ++i)
                        for (int j = 0; j < count; ++j) {
                            const double u = u0 + (i + .5) * du, v = v0 + (j + .5) * dv;
                            gp_Pnt p;
                            gp_Vec tu, tv;
                            surface.D1(u, v, p, tu, tv);
                            auto n = tu.Crossed(tv);
                            if (n.SquareMagnitude() < 1e-24) continue;
                            n.Normalize();
                            if (face.Orientation() == TopAbs_REVERSED) n.Reverse();
                            Bnd_Box box;
                            box.Add(p);
                            for (double a : {-.5, .5})
                                for (double b : {-.5, .5})
                                    box.Add(surface.Value(u + a * du, v + b * dv));
                            const bool interior = trims.Perform(gp_Pnt2d(u, v)) == TopAbs_IN;
                            Bnd_Box bound;
                            // Use native analytic or pole enclosures. Other adaptors use
                            // the complete face bounds. The five-point box remains only
                            // the original proposal filter.
                            const auto type = surface.GetType();
                            try {
                                if (type == GeomAbs_Plane || type == GeomAbs_Cylinder ||
                                    type == GeomAbs_Sphere || type == GeomAbs_BezierSurface ||
                                    type == GeomAbs_BSplineSurface)
                                    BndLib_AddSurface::Add(surface, u - .5 * du, u + .5 * du, v - .5 * dv,
                                                           v + .5 * dv, Precision::Confusion(), bound);
                                if (type == GeomAbs_BSplineSurface) {
                                    try {
                                        const auto spline =
                                            Handle(Geom_BSplineSurface)::DownCast(surface.BSpline()->Copy());
                                        double U0, U1, V0, V1;
                                        spline->Bounds(U0, U1, V0, V1);
                                        if ((spline->IsUPeriodic() ||
                                             (u - .5 * du >= U0 && u + .5 * du <= U1)) &&
                                            (spline->IsVPeriodic() ||
                                             (v - .5 * dv >= V0 && v + .5 * dv <= V1))) {
                                            spline->Segment(u - .5 * du, u + .5 * du, v - .5 * dv,
                                                            v + .5 * dv);
                                            Bnd_Box tight;
                                            for (int h = 1; h <= spline->NbUPoles(); ++h)
                                                for (int k = 1; k <= spline->NbVPoles(); ++k)
                                                    tight.Add(spline->Pole(h, k));
                                            tight.Enlarge(
                                                std::max(BRep_Tool::Tolerance(face), Precision::Confusion()));
                                            bound = tight;
                                        }
                                    } catch (const Standard_Failure&) { /* Keep the native enclosure. */
                                    }
                                }
                            } catch (const Standard_Failure&) {
                                bound.SetVoid();
                            }
                            if (bound.IsVoid() || bound.IsOpen()) {
                                bound.SetVoid();
                                for (const auto xyz : face_bounds[f - 1])
                                    bound.Add(cad::point(scale(xyz, 1000)));
                            }
                            bound.Add(box);
                            result.push_back({cad::coordinates(p),
                                              {n.X(), n.Y(), n.Z()},
                                              cad::coordinates(box.CornerMin()),
                                              cad::coordinates(box.CornerMax()),
                                              cad::coordinates(bound.CornerMin()),
                                              cad::coordinates(bound.CornerMax()),
                                              std::size_t(f),
                                              interior,
                                              u,
                                              v});
                        }
                }
                return result;
            });
        std::vector<Bnd_Box> enclosures;
        for (const auto& row : stored) {
            Chart chart{cad::point(row.point),
                        gp_Vec(row.normal[0], row.normal[1], row.normal[2]),
                        {},
                        std::size_t(row.face),
                        bool(row.interior),
                        row.u,
                        row.v};
            chart.search_box.Add(cad::point(row.lo));
            chart.search_box.Add(cad::point(row.hi));
            Bnd_Box bound;
            bound.Add(cad::point(row.bound_lo));
            bound.Add(cad::point(row.bound_hi));
            enclosures.push_back(bound);
            const auto id = charts.size();
            boxes.Add(id, chart.search_box);
            if (chart.interior) {
                Bnd_Box site;
                site.Add(chart.point);
                points.Add(id, site);
            }
            charts.push_back(chart);
        }
        points.Fill(); boxes.Fill();
        Bnd_Box all; all.SetWhole();
        Selection p(all),t(all);
        sites.Select(p); tiles.Select(t);
        point_order.resize(charts.size()); tile_order.resize(charts.size());
        for (std::size_t i=0;i<p.ids.size();++i) point_order[p.ids[i]]=i;
        for (std::size_t i=0;i<t.ids.size();++i) tile_order[t.ids[i]]=i;
        const auto hierarchy = [&](ChartTree& tree, std::vector<std::size_t>& ids, bool sites) {
            tree.nodes = assets.table<ChartTree::Node>((sites ? "sites-" : "tiles-") + std::to_string(count),
                                                       2 * ids.size(), [&] {
                                                           ChartTree built;
                                                           built.build(ids, charts, sites, enclosures);
                                                           return std::move(built.nodes);
                                                       });
        };
        hierarchy(point_index, p.ids, true);
        hierarchy(tile_index, t.ids, false);
        struct StoredNode {
            std::uint64_t face;
            AreaNode node;
        };
        const auto nodes = assets.table<StoredNode>("area12", std::size_t(faces.Extent()) * 144, [&] {
            std::vector<StoredNode> result;
            for (std::size_t i = 0; i < surfaces.size(); ++i) {
                for (const auto& node : quadrature(i + 1, 12).nodes)
                    result.push_back({i, node});
            }
            return result;
        });
        dense.resize(surfaces.size());
        base_rule.resize(surfaces.size());
        for (const auto& row : nodes)
            base_rule.at(row.face).nodes.push_back(row.node);
        for (auto& rule : base_rule)
            rule.index(column_reach);
    }
    FaceRule dense_rule(std::size_t face, int order) const {
        cad::PatchAssets cache(surfaces.at(face - 1)->surface.Face());
        FaceRule result;
        result.nodes =
            cache.table<AreaNode>("quadrature-" + std::to_string(order), std::size_t(order) * order,
                                  [&] { return quadrature(face, order).nodes; });
        result.index(column_reach);
        return result;
    }
    const FaceRule& shared_rule(std::size_t face, int order) {
        if (order == 12) return base_rule.at(face - 1);
        auto& entry = dense.at(face - 1);
        const int level = order == 24 ? 0 : order == 48 ? 1 : 2;
        std::call_once(entry.once[level], [&] { entry.rules[level] = dense_rule(face, order); });
        return entry.rules[level];
    }
    void prepare_dense() const {
        // Write one face at a time so preprocessing does not retain every
        // high-order node simultaneously.
        for (std::size_t face = 1; face <= surfaces.size(); ++face)
            for (const int order : {24, 48, 96})
                dense_rule(face, order);
    }
    FaceRule quadrature(std::size_t id, int order) const {
        FaceRule result;
        const auto& f = *surfaces[id - 1];
        const auto& s = f.surface;
        const double u0 = s.FirstUParameter(), v0 = s.FirstVParameter();
        const double du = (s.LastUParameter() - u0) / order, dv = (s.LastVParameter() - v0) / order;
        const double sign = s.Face().Orientation() == TopAbs_REVERSED ? -1. : 1.;
        for (int i = 0; i < order; ++i)
            for (int j = 0; j < order; ++j) {
                const double u = u0 + (i + .5) * du, v = v0 + (j + .5) * dv;
                if (!f.inside(u, v)) continue;
                gp_Pnt p;
                gp_Vec tu, tv;
                f.d1(u, v, p, tu, tv);
                const auto n = tu.Crossed(tv);
                result.nodes.push_back({scale(cad::coordinates(p), .001),
                                        scale(Vec{n.X(), n.Y(), n.Z()}, sign * du * dv * 1e-6)});
            }
        return result;
    }
};
struct Proposal {
    double score;
    std::size_t site,tile;
    bool reverse;
};
using FacePair=std::array<std::size_t,2>;
using Proposals=std::map<FacePair,std::vector<Proposal>>;

// Each table bounds a neighbourhood of one relative rigid transformation.
struct ChartWindow {
    gp_Trsf anchor;
    std::size_t bin;
    std::vector<unsigned char> leaves;
    void append(std::size_t delta) {
        while (delta>=128) {
            leaves.push_back(static_cast<unsigned char>((delta&127)|128));delta>>=7;
        }
        leaves.push_back(static_cast<unsigned char>(delta));
    }
    std::size_t read(std::size_t& offset) const {
        std::size_t value=0;
        for (unsigned shift=0;;shift+=7) {
            const auto byte=leaves[offset++];
            value|=std::size_t(byte&127)<<shift;
            if (!(byte&128)) return value;
        }
    }
};
struct CandidateWindow {
    std::deque<ChartWindow> saved;
    std::unordered_multimap<std::size_t,const ChartWindow*> bins;
    const ChartWindow* active{};
    std::size_t bytes{};
    static std::size_t key(const Box& box,const gp_Trsf& pose,double margin) {
        std::size_t bin=0;
        const auto mix=[&](double x) {
            bin ^= std::hash<double>{}(std::floor(x/(8*margin))) + 0x9e3779b9u + (bin<<6) + (bin>>2);
        };
        const auto center=cad::point(box.center).Transformed(pose);
        for (int i=1;i<=3;++i) {
            mix(center.Coord(i));
            for (int j=1;j<=3;++j) mix(pose.Value(i,j)*box.half[j-1]);
        }
        return bin;
    }
    void insert(ChartWindow window) {
        constexpr std::size_t budget=32u*1024u*1024u;
        const auto size=[](const ChartWindow& x) {
            return sizeof(ChartWindow)+x.leaves.capacity()*sizeof(x.leaves.front())+64;
        };
        const auto extra=size(window);
        while (!saved.empty() && bytes+extra>budget) {
            const auto& old=saved.front();
            const auto [first,last]=bins.equal_range(old.bin);
            const auto it=std::find_if(first,last,[&](const auto& item){return item.second==&old;});
            bins.erase(it);
            bytes-=size(old);
            saved.pop_front();
        }
        saved.push_back(std::move(window));
        active=&saved.back();
        bins.emplace(active->bin,active);
        bytes+=extra;
    }
};
// Bound displacement in each source-body coordinate for all points of the
// target root box. Also bounds movement of every transformed child AABB.
double box_motion(const Box& box, const gp_Trsf& a, const gp_Trsf& b) {
    const auto center = cad::point(box.center);
    const auto shift = center.Transformed(a).XYZ() - center.Transformed(b).XYZ();
    double largest = 0.;
    for (int i = 1; i <= 3; ++i) {
        double delta = std::abs(shift.Coord(i));
        for (int j = 1; j <= 3; ++j)
            delta += std::abs(a.Value(i, j) - b.Value(i, j)) * box.half[j - 1];
        largest = std::max(largest, delta);
    }
    return largest;
}
bool collect_indexed(const Atlas& source, const Atlas& target, const gp_Trsf& pose, double reach,
                     bool reverse, Proposals& candidates, CandidateWindow& cache) {
    const auto& sites=source.point_index.nodes;
    const auto& tiles=target.tile_index.nodes;
    if (sites.empty() || tiles.empty()) return false;
    const auto& root = tiles.front().box;
    const double margin=reach;
    const double turn=std::min(2.,margin/std::max(reach,length(root.half)));
    const auto valid=[&](const gp_Trsf& anchor) {
        if (box_motion(root,pose,anchor)>margin) return false;
        double norm2=0;
        for (int i=1;i<=3;++i) for (int j=1;j<=3;++j) {
            const double d=pose.Value(i,j)-anchor.Value(i,j);norm2+=d*d;
        }
        return norm2<=2*turn*turn;
    };
    bool built=false;
    if (!cache.active || !valid(cache.active->anchor)) {
        const auto bin=CandidateWindow::key(root,pose,margin);
        const auto [first,last]=cache.bins.equal_range(bin);
        cache.active=nullptr;
        for (auto it=first;it!=last;++it) {
            if (valid(it->second->anchor)) {cache.active=it->second;break;}
        }
        if (!cache.active) {
            ChartWindow window{pose,bin,{}};
            std::vector<std::pair<std::size_t,std::size_t>> leaves;
            const auto matrix=pose.VectorialPart();
            const double padding=reach+margin+Precision::Confusion()+
                64*std::numeric_limits<double>::epsilon()*
                (1+length(root.center)+length(root.half)+pose.TranslationPart().Modulus());
            const auto visit_tile=[&](auto&& self,int t,int s)->void {
                const auto& tile=tiles[t];
                Box box;
                box.center=cad::coordinates(cad::point(tile.box.center).Transformed(pose));
                for (int i=0;i<3;++i) {
                    box.half[i]=padding;
                    for (int j=0;j<3;++j) box.half[i]+=std::abs(matrix.Value(i+1,j+1))*tile.box.half[j];
                }
                const auto visit_sites=[&](auto&& descend,int k)->void {
                    const auto& site=sites[k];
                    if (!box.overlaps(site.box)) return;
                    if (tile.left>=0 && (site.left<0 || squared(tile.box.half)>squared(site.box.half))) {
                        self(self,tile.left,k);self(self,tile.right,k);
                    } else if (site.left>=0) {
                        descend(descend,site.left);descend(descend,site.right);
                    } else {
                        const auto& f=target.charts[tile.chart];
                        const auto& p=source.charts[site.chart];
                        auto tight=f.search_box.Transformed(pose);
                        tight.SetGap(tight.GetGap()+padding);
                        if (tight.IsOut(p.point)) return;
                        const auto n=f.normal.Transformed(pose);
                        const auto v=gp_Vec(f.point.Transformed(pose),p.point);
                        // Within this window a point moves by at most sqrt(3)*margin
                        // and a unit normal by at most turn (spectral rotation norm).
                        if (-n.Dot(p.normal)+turn+1e-12<.5 ||
                            std::abs(v.Dot(n))>reach+std::sqrt(3.)*margin+v.Magnitude()*turn+Precision::Confusion()) return;
                        leaves.emplace_back(tile.chart,site.chart);
                    }
                };
                visit_sites(visit_sites,s);
            };
            visit_tile(visit_tile,0,0);
            std::sort(leaves.begin(),leaves.end());
            std::size_t tile=0,site=0;
            for (auto [t,s]:leaves) {
                window.append(t-tile);
                window.append(t==tile ? s-site : s);
                tile=t;site=s;
            }
            cache.insert(std::move(window));
            built=true;
        }
    }
    auto source_box=source.body_box.Transformed(pose.Inverted());
    source_box.Enlarge(reach);
    std::size_t last=std::numeric_limits<std::size_t>::max();
    gp_Pnt origin;
    gp_Vec outward;
    Bnd_Box exact;
    std::size_t t=0,s=0;
    for (std::size_t offset=0;offset<cache.active->leaves.size();) {
        const auto delta=cache.active->read(offset);
        t+=delta;
        s=(delta ? 0 : s)+cache.active->read(offset);
        const auto& tile=target.charts[t];
        if (last!=t) {
            last=t;
            exact.SetVoid();
            if (!source_box.IsOut(tile.search_box)) {
                exact=tile.search_box.Transformed(pose);
                exact.Enlarge(reach);
                origin=tile.point.Transformed(pose);
                outward=tile.normal.Transformed(pose);
            }
        }
        const auto& site=source.charts[s];
        if (exact.IsVoid() || exact.IsOut(site.point)) continue;
        const double cosine=-site.normal.Dot(outward);
        if (cosine<.5) continue;
        const double d=-gp_Vec(origin,site.point).Dot(outward)/cosine;
        if (std::abs(d)>reach) continue;
        const FacePair pair=reverse ? FacePair{tile.face,site.face} : FacePair{site.face,tile.face};
        candidates[pair].push_back({d,s,t,reverse});
    }
    return built;
}

void collect(const Atlas& source,const Atlas& target,const gp_Trsf& target_in_source,
             double reach_mm,bool reverse,Proposals& candidates) {
    auto window=source.body_box.Transformed(target_in_source.Inverted());
    window.Enlarge(reach_mm);
    Selection tiles(window);
    target.tiles.Select(tiles);
    for (const auto id:tiles.ids) {
        const auto& tile=target.charts[id];
        auto box=tile.search_box.Transformed(target_in_source);
        box.Enlarge(reach_mm);
        Selection sites(box);
        source.sites.Select(sites);
        if (sites.ids.empty()) continue;
        const auto origin=tile.point.Transformed(target_in_source);
        const auto outward=tile.normal.Transformed(target_in_source);
        for (auto s:sites.ids) {
            const auto& site=source.charts[s];
            const double cosine=-site.normal.Dot(outward);
            if (cosine<.5) continue;
            const double estimated=-gp_Vec(origin,site.point).Dot(outward)/cosine;
            if (std::abs(estimated)>reach_mm) continue;
            const FacePair pair=reverse ? FacePair{tile.face,site.face} : FacePair{site.face,tile.face};
            candidates[pair].push_back({estimated,s,id,reverse});
        }
    }
}

// Intersect the supporting surface near the candidate chart in millimetres.
// This inexpensive proposal may fail at folds. The accepted overlap still
// requires both nearest oriented, trim-valid exits from the full CAD face tree.
std::optional<double> propose_depth(const Atlas& target,const Chart& tile,
                                    gp_Pnt p,gp_Vec direction,double reach,bool require_exit=false) {
    const auto& face=*target.surfaces[tile.face-1];
    const auto& surf=face.surface;
    double u=tile.u,v=tile.v;
    for (int iter=0;iter<8;++iter) {
        gp_Pnt q;
        gp_Vec du,dv;
        face.d1(u,v,q,du,dv);
        const auto delta=gp_Vec(p,q);
        const double t=delta.Dot(direction);
        const auto residual=delta-direction*t;
        if (residual.SquareMagnitude()<1e-14) {
            if (std::abs(t)>reach || !face.inside(u,v)) return {};
            if (require_exit) {
                const auto normal=du.Crossed(dv);
                const double sign=surf.Face().Orientation()==TopAbs_REVERSED ? -1. : 1.;
                if (sign*direction.Dot(normal)<=1e-8*normal.Magnitude()) return {};
            }
            return t;
        }
        const auto c=dv.Crossed(direction),e=direction.Crossed(du);
        const double det=du.Dot(c);
        if (std::abs(det)<1e-11*du.Magnitude()*dv.Magnitude() || det==0) return {};
        u-=residual.Dot(c)/det;
        v-=residual.Dot(e)/det;
        if (!(u>=surf.FirstUParameter() && u<=surf.LastUParameter()
              && v>=surf.FirstVParameter() && v<=surf.LastVParameter())) return {};
    }
    return {};
}
struct Seed {
    Proposal proposal;
    Chart site;
};
struct Search {
    Atlas& a;
    Atlas& b;
    CadExitQuery &ray_a,&ray_b;
    const std::array<ContactPose,2>& poses;
    double reach;
    int grid;
    std::size_t tests{},attempts{};
    std::array<gp_Trsf,2> world,relative;
    Search(Atlas& a,Atlas& b,CadExitQuery& ray_a,CadExitQuery& ray_b,
           const std::array<ContactPose,2>& poses,double reach,int grid):
        a(a),b(b),ray_a(ray_a),ray_b(ray_b),poses(poses),reach(reach),grid(grid),
        world{cad::placement(poses[0]),cad::placement(poses[1])},
        relative{world[1].Inverted()*world[0],world[0].Inverted()*world[1]} {}

    double evaluate(const Seed& seed, FacePair pair, std::optional<CadChord>& hit) {
        const bool reverse=seed.proposal.reverse;
        const auto& other=reverse ? a : b;
        const auto& site=seed.site;
        const auto local=site.point.Transformed(relative[reverse]);
        const auto dir=-site.normal.Transformed(relative[reverse]);
        ++attempts;
        const auto depth=propose_depth(other,other.charts[seed.proposal.tile],local,dir,reach*1000);
        if (!depth) return -infinity;
        if (*depth>2*Precision::Confusion()) {
            const Vec p=scale(cad::coordinates(site.point.Transformed(world[reverse])),.001);
            const auto outward=site.normal.Transformed(world[reverse]);
            const Vec n{outward.X(),outward.Y(),outward.Z()};
            auto chord=local_chord(ray_a,ray_b,poses,sub(p,scale(n,.0005 * *depth)),
                                   scale(n,reverse ? -1. : 1.),reach,pair,tests);
            if (chord && chord->exits[0].face==pair[0] && chord->exits[1].face==pair[1]) hit=chord;
        }
        return *depth;
    }

    // Follow a near miss in the native parameter domain. Only a verified
    // positive chord stops the search. Searching a finite stencil is heuristic.
    void refine(Seed& best,double score,FacePair pair,std::optional<CadChord>& hit) {
        const auto& body=best.proposal.reverse ? b : a;
        const auto& face=*body.surfaces[best.site.face-1];
        const auto& surf=face.surface;
        double du=(surf.LastUParameter()-surf.FirstUParameter())/(2*grid);
        double dv=(surf.LastVParameter()-surf.FirstVParameter())/(2*grid);
        const double grazing_band=std::sqrt(Precision::Confusion()*1000*reach);
        for (int k=0;k<10 && !hit;++k,du*=.5,dv*=.5) {
            // A nearly tangent pair can contain a loaded interior between
            // the last two stencil points. Refine only these near misses.
            if (k>=5 && score < -grazing_band) break;
            const auto center=best.site;
            for (const auto offset:std::array<std::array<int,2>,8>{{{1,0},{-1,0},{0,1},{0,-1},{1,1},{-1,1},{1,-1},{-1,-1}}}) {
                Seed trial=best;
                auto& s=trial.site;
                s.u=center.u+du*offset[0]; s.v=center.v+dv*offset[1];
                if (s.u<surf.FirstUParameter() || s.u>surf.LastUParameter() || s.v<surf.FirstVParameter() ||
                    s.v>surf.LastVParameter() || !face.inside(s.u,s.v)) continue;
                gp_Vec tu,tv;
                face.d1(s.u,s.v,s.point,tu,tv);
                s.normal=tu.Crossed(tv);
                if (s.normal.SquareMagnitude()<1e-24) continue;
                s.normal.Normalize();
                if (surf.Face().Orientation()==TopAbs_REVERSED) s.normal.Reverse();
                const auto depth=evaluate(trial,pair,hit);
                if (depth>score || hit) {best=trial; score=depth;}
                if (hit) break;
            }
        }
    }
};

// Midpoint surface quadrature on the original UV charts. Each material-side
// face contributes once per group, including when several opposing faces
// cover it. The opposing trim and positive thickness clip its loaded part.
struct SurfaceQuadrature {
    Atlas& body;
    int order;
    std::map<std::size_t, FaceRule> saved;
    std::size_t skipped{};
    const FaceRule& rule(std::size_t id) {
        if (order == 12 || order == 24 || order == 48 || order == 96) return body.shared_rule(id, order);
        auto [it,fresh]=saved.try_emplace(id);
        if (fresh) {
            it->second = body.quadrature(id, order);
            it->second.index(body.column_reach);
        }
        return it->second;
    }
};
struct PressurePatch {Vec center,normal,arm;double depth;};
std::optional<PressurePatch> integrate_patch(std::array<SurfaceQuadrature,2>& grids,
    CadExitQuery& ray_a,CadExitQuery& ray_b,const std::array<ContactPose,2>& poses,
    const std::vector<CadChord>& group,double reach,cad::PressureFailure* failure) {
    Vec area{},moment{},centroid{};
    const Vec origin=poses[0].origin;
    double volume=0;
    for (int side=0;side<2;++side) {
        const int other=1-side;
        const double sign=side ? -1. : 1.;
        const auto world=cad::placement(poses[side]);
        const auto to_target=cad::placement(poses[other]).Inverted()*world;
        // The discovery chords identify the faces in this region. The material
        // normal at each integration point defines its column independently.
        std::map<std::size_t,std::set<std::size_t>> faces;
        for (const auto& w:group) {
            faces[w.exits[side].face].insert(w.exits[other].face);
        }
        for (const auto& [id, face_set] : faces) {
            const std::vector<std::size_t> targets(face_set.begin(),face_set.end());
            std::vector<std::vector<std::size_t>> spans(targets.size());
            std::vector<std::size_t> active;
            const auto& rule = grids[side].rule(id);
            for (const auto& block : rule.blocks) {
                const auto swept=block.placed(to_target);
                const Bnd_Box bounds(gp_Pnt(swept.lo),gp_Pnt(swept.hi));
                active.clear();
                for (std::size_t i=0;i<targets.size();++i) {
                    const auto face=targets[i]-1;
                    if (bounds.IsOut(grids[other].body.face_boxes[face])) continue;
                    const auto& surface=*grids[other].body.surfaces[face];
                    surface.block_spans(swept,spans[i]);
                    if (surface.hulls.empty() || !spans[i].empty()) active.push_back(i);
                }
                if (active.empty()) {
                    grids[side].skipped += block.end - block.begin;
                    continue;
                }
                for (auto j = block.begin; j < block.end; ++j) {
                    const auto& sample = rule.nodes[j];
                    const auto native=cad::point(scale(sample.point,1000));
                    const auto local = native.Transformed(to_target);
                    const auto weight=length(sample.area);
                    if (!(weight>0)) continue;
                    const auto ray=poses[side].rotate(scale(sample.area,-1/weight));
                    const auto direction=poses[other].inverse_rotate(ray);
                    const gp_XYZ segment(direction[0]*1000*reach,direction[1]*1000*reach,direction[2]*1000*reach);
                    const auto hint=std::find_if(active.begin(),active.end(),[&](auto i){
                        return grids[other].body.surfaces[targets[i]-1]->column_possible(local.XYZ(),segment,&spans[i]);
                    });
                    if (hint==active.end()) continue;
                    // Keep the native face visitation order for coincident
                    // hits, including conservative hints outside this block.
                    const auto first=std::find_if(targets.begin(),targets.begin()+*hint,[&](auto face) {
                        return grids[other].body.surfaces[face-1]->column_possible(local.XYZ(),segment);
                    });
                    std::size_t tests=0;
                    const auto hit=(other ? ray_b : ray_a).first_exit(
                        scale(cad::coordinates(local),.001),direction,reach,*first,tests,targets);
                    if (!hit.face || !(hit.distance_m>0)) continue;
                    const double depth=hit.distance_m;
                    const auto point = scale(cad::coordinates(native.Transformed(world)), .001);
                    const auto area_element = scale(poses[side].rotate(sample.area), sign);
                    const auto r = sub(point, origin);
                    area = add(area, area_element);
                    moment = add(moment, cross3(r, area_element));
                    const double dv = depth * weight;
                    centroid = add(centroid, scale(add(r, scale(ray, .5 * depth)), dv));
                    volume += dv;
                }
            }
        }
    }
    const double magnitude=length(area);
    if (!(volume>0 && magnitude>0)) {
        if (failure) *failure=volume>0 ? cad::PressureFailure::no_resultant : cad::PressureFailure::no_columns;
        return {};
    }
    const auto center=add(origin,scale(centroid,1/volume));
    const auto normal=scale(area,1/magnitude);
    std::size_t tests=0;
    auto chord=local_chord(ray_a,ray_b,poses,center,normal,reach,{},tests);
    if (!chord) {
        if (failure) *failure=cad::PressureFailure::centroid_chord;
        return {};
    }
    if (failure) *failure=cad::PressureFailure::none;
    return PressurePatch{center,normal,sub(scale(moment,1/magnitude),cross3(sub(center,origin),normal)),
                         chord->region.depth_m};
}


}
namespace nc::cad {
using namespace dynamics;
struct PatchSearch::Data {
    Atlas a,b;
    int grid;
    double reach;
    std::array<CandidateWindow, 2> windows;
    using EdgeKey = std::array<FacePair, 2>;
    std::map<EdgeKey, ChordPath> paths;
    std::map<FacePair,Seed> previous;
    struct Worker {
        CadExitQuery a,b;
        std::map<int,std::array<SurfaceQuadrature,2>> levels;
        Worker(const Atlas& x,const Atlas& y):a(x.index,x.bezier),b(y.index,y.bezier) {}
    };
    static constexpr int worker_limit=4;
    std::array<std::unique_ptr<Worker>,worker_limit> workers;
    Data(const TopoDS_Shape& first,const TopoDS_Shape& second,int grid,double reach,bool bezier):
        a(first,grid,bezier,reach),b(second,grid,bezier,reach),grid(grid),reach(reach) {
        for (auto& w:workers) w=std::make_unique<Worker>(a,b);
    }
    PatchDiscovery discover(const std::array<ContactPose,2>& poses,bool indexed,bool group) {
        const auto t=Clock::now();
        const auto b_in_a=cad::placement(poses[0]).Inverted()*cad::placement(poses[1]);
        std::array<Proposals,2> sides;
        std::array<int, 2> builds{}, queries{};
        OSD_ThreadPool::Launcher pool(*OSD_ThreadPool::DefaultPool(),indexed ? worker_limit : 1);
        pool.Perform(0, 2, [&](int, int side) {
            const auto relative = side ? b_in_a.Inverted() : b_in_a;
            const auto& source = side ? b : a;
            const auto& target = side ? a : b;
            if (indexed) {
                queries[side] = !source.point_index.nodes.empty() && !target.tile_index.nodes.empty();
                builds[side] =
                    collect_indexed(source, target, relative, 1000 * reach, side, sides[side], windows[side]);
            } else
                collect(source, target, relative, 1000 * reach, side, sides[side]);
        });
        auto& proposals=sides[0];
        for (const auto& [pair,sites]:sides[1]) {
            auto& dest=proposals[pair]; dest.insert(dest.end(),sites.begin(),sites.end());
        }
        if (indexed) for (auto& [pair,sites]:proposals) {
            const auto key=[&](const Proposal& s) {
                return std::tuple{s.reverse,(s.reverse?a:b).tile_order[s.tile],(s.reverse?b:a).point_order[s.site]};
            };
            std::sort(sites.begin(),sites.end(),[&](const auto& l,const auto& r){return key(l)<key(r);});
        }
        for (const auto& [pair,seed]:previous) proposals.try_emplace(pair);
        const auto shortlisted=Clock::now();
        struct Job {
            Proposals::iterator where;
            std::optional<CadChord> hit;
            std::optional<Seed> best;
            std::size_t tracked{},attempts{},tests{};
        };
        std::vector<Job> jobs;
        for (auto it=proposals.begin();it!=proposals.end();++it) jobs.push_back({it,{},{}});
        pool.Perform(0,static_cast<int>(jobs.size()),[&](int idx,int n) {
            auto& job=jobs[n];
            auto& [pair,sites]=*job.where;
            auto& hit=job.hit;
            auto& best=job.best;
            auto& w=*workers[idx];
            Search search{a,b,w.a,w.b,poses,reach,grid};
            double score=-infinity;
            const auto probe=[&](const Seed& seed) {
                const double d=search.evaluate(seed,pair,hit);
                if (d>score || hit) {score=d; best=seed;}
            };
            if (const auto old=previous.find(pair);old!=previous.end()) {
                probe(old->second);
                job.tracked=hit.has_value();
            }
            std::sort(sites.begin(),sites.end(),[](const auto& l,const auto& r){return l.score>r.score;});
            std::vector<std::pair<bool,std::size_t>> tried;
            for (const auto& s:sites) {
                if (hit) break;
                const auto key=std::pair{s.reverse,s.site};
                if (std::find(tried.begin(),tried.end(),key)!=tried.end()) continue;
                tried.push_back(key);
                probe({s,(s.reverse ? b : a).charts[s.site]});
            }
            if (!hit && best) search.refine(*best,score,pair,hit);
            job.attempts=search.attempts;job.tests=search.tests;
        });
        std::map<FacePair,Seed> next;
        std::vector<std::optional<CadChord>> found;
        std::size_t attempts=0,tracked=0,tests=0;
        for (const auto& job:jobs) {
            if (job.hit) {found.push_back(job.hit); next.emplace(job.where->first,*job.best);}
            attempts+=job.attempts; tracked+=job.tracked; tests+=job.tests;
        }
        previous=std::move(next);
        const auto search_end=Clock::now();
        std::vector<int> labels;
        int reused_paths = 0;
        if (group && found.size()>=3 && found.size()<=24 && pool.HasThreads()) {
            std::vector<std::pair<int,int>> edges;
            for (int i=1;i<static_cast<int>(found.size());++i)
                for (int j=0;j<i;++j) edges.emplace_back(i,j);
            std::vector<ChordPath> hints(edges.size());
            const auto key = [&](int i, int j) {
                const auto faces = [&](int k) {
                    return FacePair{found[k]->exits[0].face, found[k]->exits[1].face};
                };
                return EdgeKey{faces(i), faces(j)};
            };
            if (!paths.empty()) {
                for (std::size_t k = 0; k < edges.size(); ++k) {
                    const auto [i, j] = edges[k];
                    if (auto it = paths.find(key(i, j)); it != paths.end()) hints[k] = std::move(it->second);
                }
            }
            std::vector<unsigned char> connected(edges.size());
            pool.Perform(0,static_cast<int>(edges.size()),[&](int idx,int k) {
                const auto [i,j]=edges[k]; auto& w=*workers[idx];
                connected[k] = connected_chords(w.a, w.b, poses, *found[i], *found[j], reach, &hints[k]);
            });
            paths.clear();
            for (std::size_t k = 0; k < hints.size(); ++k) {
                reused_paths += hints[k].reused;
                if (!hints[k].intermediate_in_a.empty()) {
                    const auto [i, j] = edges[k];
                    paths.emplace(key(i, j), std::move(hints[k]));
                }
            }
            labels.resize(found.size());std::iota(labels.begin(),labels.end(),0);
            for (std::size_t k=0;k<edges.size();++k) if (connected[k]) {
                const auto [i,j]=edges[k]; const int old=labels[i],replacement=labels[j];
                std::replace(labels.begin(),labels.end(),old,replacement);
            }
        } else if (group) labels=chord_groups(a.rays,b.rays,poses,found,reach);
        const auto end=Clock::now();
        const double seconds=std::chrono::duration<double>(end-t).count();

        return {std::move(found),
                std::move(labels),
                proposals.size(),
                attempts,
                tracked,
                tests,
                seconds,
                std::chrono::duration<double>(shortlisted - t).count(),
                std::chrono::duration<double>(end - search_end).count(),
                builds[0] + builds[1],
                queries[0] + queries[1] - builds[0] - builds[1],
                reused_paths};
    }
};
void prepare_patch_part(const TopoDS_Shape& shape,bool bezier) {
    Atlas part(shape, 6,bezier);
    part.prepare_dense();
}
PatchSearch::PatchSearch(const TopoDS_Shape& a,const TopoDS_Shape& b,int grid,double reach,bool bezier):
    data_(std::make_unique<Data>(a,b,grid,reach,bezier)) {}
PatchSearch::~PatchSearch()=default;
void PatchSearch::prepare_dense() const {
    data_->a.prepare_dense();
    data_->b.prepare_dense();
}
std::array<std::size_t, 2> PatchSearch::asset_counts() const {
    return {data_->a.assets.loaded + data_->b.assets.loaded, data_->a.assets.built + data_->b.assets.built};
}
std::size_t PatchSearch::skipped_nodes() const {
    std::size_t count = 0;
    for (const auto& worker : data_->workers)
        for (const auto& [order, levels] : worker->levels)
            for (const auto& level : levels)
                count += level.skipped;
    return count;
}
std::array<std::size_t,4> PatchSearch::ray_counts() const {
    std::array<std::size_t,4> result{};
    const auto add=[&](const CadExitQuery& rays) {
        const auto counts=rays.ray_counts();
        for (std::size_t i=0;i<result.size();++i) result[i]+=counts[i];
    };
    add(data_->a.rays); add(data_->b.rays);
    for (const auto& w:data_->workers) {add(w->a);add(w->b);}
    return result;
}
bool PatchSearch::boxes_separate(const std::array<ContactPose,2>& poses) const {
    auto a=data_->a.body_box.Transformed(placement(poses[0]));
    a.Enlarge(a.GetGap()+2*Precision::Confusion());
    return a.IsOut(data_->b.body_box.Transformed(placement(poses[1])));
}
PatchDiscovery PatchSearch::discover(const std::array<ContactPose,2>& poses,bool indexed,bool group) {
    return data_->discover(poses,indexed,group);
}
std::optional<CadChord> PatchSearch::chord(const std::array<ContactPose,2>& poses,Vec p,Vec n,std::array<std::size_t,2> ids) {
    std::size_t tests=0;
    return local_chord(data_->a.rays,data_->b.rays,poses,p,n,data_->reach,ids,tests);
}
std::optional<ContactRegion> PatchSearch::integrate(const std::array<ContactPose,2>& poses,
                                                    const std::vector<CadChord>& group,int order,int worker,
                                                    PressureFailure* failure) {
    if (order==0) {
        for (int level:{12,24,48,96}) if (auto r=integrate(poses,group,level,worker,failure)) return r;
        return {};
    }
    auto& w=*data_->workers.at(worker);
    auto& levels=w.levels;
    if (!levels.contains(order)) levels.emplace(order,std::array<SurfaceQuadrature,2>{{{data_->a,order,{}},{data_->b,order,{}}}});
    const auto p=integrate_patch(levels.at(order),w.a,w.b,poses,group,data_->reach,failure);
    if (!p) return {};
    return ContactRegion{p->center,p->normal,p->depth,p->arm};
}
}

namespace nc {
namespace {
using namespace dynamics;
class PatchRegionQuery final : public PairRegionQuery {
public:
    PatchRegionQuery(const TopoDS_Shape& a,const TopoDS_Shape& b,std::array<Vec,2> centers,double maximum,bool bezier):
        a_(a),b_(b),search_(a,b,6,2*maximum,bezier),centers_(centers),maximum_(maximum) {}
    bool independent_cad_patch() const override {return true;}
    std::vector<ContactRegion> regions(ContactPose a,ContactPose b,ContactWork& work) const override {
        const std::array poses{a,b};
        ++work.cad_patch_queries;
        if (search_.boxes_separate(poses)) return {};
        const auto search=search_.discover(poses,true,true);
        work.cad_window_builds += search.window_builds;
        work.cad_window_reuses += search.window_reuses;
        work.cad_path_reuses += search.reused_paths;
        work.cad_patch_broad_s+=search.broad_s;
        work.cad_patch_search_s+=search.query_s-search.group_s-search.broad_s;
        work.cad_patch_group_s+=search.group_s;
        if (search.witnesses.empty()) {
            const auto tic=Clock::now();
            bool separated=false;
            try {
                if (!boundary_a_) boundary_a_=cad::boundary_index(a_);
                if (!boundary_b_) boundary_b_=cad::boundary_index(b_);
                const auto relative=cad::placement(b).Inverted()*cad::placement(a);
                SurfaceTransition stats;
                const double tol=2*Precision::Confusion();
                separated=cad::bounded_distance(boundary_a_,boundary_b_,relative,4*tol,tol,stats,&witness_)>tol;
            } catch (const UnresolvedContact&) { /* Resolve directly on the placed solids. */ }
              catch (const Standard_Failure&) {}
            work.cad_patch_search_s+=std::chrono::duration<double>(Clock::now()-tic).count();
            if (separated) return {};
            return boolean_pressure(poses,work,false);
        }
        std::map<int,std::vector<CadChord>> groups;
        for (std::size_t i=0;i<search.labels.size();++i) groups[search.labels[i]].push_back(*search.witnesses[i]);
        const auto start=Clock::now();
        std::vector<const std::vector<CadChord>*> components;
        for (const auto& [label,group]:groups) components.push_back(&group);
        std::vector<std::optional<ContactRegion>> patches(groups.size());
        const auto previous_skips = search_.skipped_nodes();
        if (!groups.empty()) {
            OSD_ThreadPool::Launcher pool(*OSD_ThreadPool::DefaultPool(),std::min(4,static_cast<int>(groups.size())));
            pool.Perform(0,static_cast<int>(groups.size()),[&](int worker,int i) {
                patches[i]=search_.integrate(poses,*components[i],0,worker);
            });
        }
        work.cad_skipped_area_nodes += search_.skipped_nodes() - previous_skips;
        std::vector<ContactRegion> result;
        for (const auto& patch:patches) {
            if (!patch) {
                work.cad_patch_area_s+=std::chrono::duration<double>(Clock::now()-start).count();
                return boolean_pressure(poses,work,true);
            }
            result.push_back(*patch);
        }
        work.cad_patch_area_s+=std::chrono::duration<double>(Clock::now()-start).count();
        return result;
    }
    SurfaceTransition transition(ContactPose a,ContactPose b,ContactPose c,ContactPose d,double eps) const override {
        return events().transition(a,b,c,d,eps);
    }
    SurfaceTransition transition(ContactPose a,ContactPose b,ContactPose c,ContactPose d,double eps,double dt) const override {
        return events().transition(a,b,c,d,eps,dt);
    }
private:
    std::vector<ContactRegion> boolean_pressure(const std::array<ContactPose,2>& poses,ContactWork& work,bool witnessed) const {
        const auto ta=cad::placement(poses[0]),tb=cad::placement(poses[1]);
        const auto begin=Clock::now();
        std::optional<cad::LocalSolidPair> selected;
        try {
            if (!local_) local_=std::make_unique<cad::LocalSolidSearch>(a_,b_);
            selected=local_->select(tb.Inverted()*ta,100*Precision::Confusion());
        } catch (const UnresolvedContact&) {} catch (const Standard_Failure&) {}
        work.cad_region_select_s+=std::chrono::duration<double>(Clock::now()-begin).count();
        if (selected && selected->separated && !witnessed) return {};
        if (selected && !selected->separated && (!selected->a.shape.IsSame(a_) || !selected->b.shape.IsSame(b_))) {
            ++work.cad_pressure_clipped;
            try {
                auto regions=pressure_on(selected->a.shape,selected->b.shape,ta,tb,poses,work);
                if (!regions.empty() || !witnessed) return regions;
            } catch (const UnresolvedContact&) { /* Recheck on the original topology. */ }
        }
        ++work.cad_pressure_full;
        return pressure_on(a_,b_,ta,tb,poses,work);
    }
    std::vector<ContactRegion> pressure_on(const TopoDS_Shape& a,const TopoDS_Shape& b,
        const gp_Trsf& ta,const gp_Trsf& tb,const std::array<ContactPose,2>& poses,ContactWork& work) const {
        std::vector<Vec> arms;
        auto result=cad::solid_regions(a.Moved(TopLoc_Location(ta)),b.Moved(TopLoc_Location(tb)),work,false,&arms,true);
        for (std::size_t i=0;i<result.size();++i) {
            auto chord=search_.chord(poses,result[i].point,result[i].normal,{});
            if (chord) result[i].depth_m=chord->region.depth_m;
            else result[i].depth_m=limiting_depth(poses,result[i]);
            result[i].pressure_arm=arms[i];
        }
        return result;
    }
    double limiting_depth(const std::array<ContactPose,2>& poses,const ContactRegion& patch) const {
        // At an isolated chart pole the centroid ray can terminate on a
        // degenerate boundary. Check the two-sided limit on two shrinking
        // rings. A missing sample or a depth jump leaves the pose unresolved.
        const Vec n=patch.normal;
        Vec u=cross3(n,std::abs(n[0])<.75 ? Vec{1,0,0} : Vec{0,1,0});
        u=scale(u,1/length(u));
        const Vec v=cross3(n,u);
        for (double phase:{.371,1.083}) {
            double lo=infinity,hi=0,sum=0;
            int hits=0;
            for (double radius:{2e-6,1e-6}) for (int j=0;j<4;++j) {
                const double angle=phase+j*1.5707963267948966;
                const auto offset=add(scale(u,radius*std::cos(angle)),scale(v,radius*std::sin(angle)));
                const auto c=search_.chord(poses,add(patch.point,offset),n,{});
                if (!c) continue;
                const double depth=c->region.depth_m;
                lo=std::min(lo,depth);hi=std::max(hi,depth);
                if (radius==1e-6) sum+=depth;
                ++hits;
            }
            if (hits==8 && hi-lo<=std::max(2e-9,1e-5*lo)) return sum*.25;
        }
        throw UnresolvedContact("unresolved pressure depth in STEP overlap");
    }
    TopoDS_Shape a_,b_;
    mutable std::shared_ptr<const cad::BoundaryIndex> boundary_a_,boundary_b_;
    mutable cad::BoundaryWitness witness_;
    mutable std::unique_ptr<cad::LocalSolidSearch> local_;
    mutable cad::PatchSearch search_;
    std::array<Vec,2> centers_;
    double maximum_;
    mutable std::once_flag event_init_;
    mutable std::shared_ptr<const PairRegionQuery> events_;
    const PairRegionQuery& events() const {
        std::call_once(event_init_,[&]{events_=cad_local_region_query(a_,b_,centers_,maximum_);});
        return *events_;
    }
};
}
std::shared_ptr<const dynamics::PairRegionQuery> cad_patch_region_query(
    const TopoDS_Shape& a,const TopoDS_Shape& b,std::array<Vec,2> centers,double maximum,bool bezier) {
    if (!(maximum>0 && std::isfinite(maximum))) throw std::invalid_argument("invalid maximum depth");
    return std::make_shared<PatchRegionQuery>(a,b,centers,maximum,bezier);
}
}
