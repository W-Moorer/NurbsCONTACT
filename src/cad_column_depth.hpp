#pragma once
#include <gp_XYZ.hxx>
#include <Precision.hxx>
#include <algorithm>
#include <array>
#include <cmath>
#include <span>

namespace nc::cad {
struct MaterialRay { gp_XYZ point, direction; };
struct ScalarRange {
    double lo, hi;
    void include(double x) { lo=std::min(lo,x); hi=std::max(hi,x); }
    double magnitude() const { return std::max(std::abs(lo),std::abs(hi)); }
};

// In these coordinates the control net is (u,v,0) plus a bounded
// displacement. Bernstein differences enclose its first derivatives too.
struct ColumnDepthBound {
    using Net=std::array<std::array<gp_XYZ,4>,4>;
    gp_XYZ origin;
    std::array<gp_XYZ,3> row{};
    std::array<double,3> error{}, gradient{};
    bool regular{};
    ColumnDepthBound()=default;
    explicit ColumnDepthBound(const Net& net,double tolerance=Precision::Confusion()):origin(net[0][0]),tol(tolerance) {
        const auto a=net[3][0]-origin,b=net[0][3]-origin,n=a.Crossed(b);
        const double square=n.SquareModulus();
        if (!(square>1e-20*a.SquareModulus()*b.SquareModulus())) return;
        row={b.Crossed(n)/square,n.Crossed(a)/square,n/std::sqrt(square)};
        for (int k=0;k<3;++k) {
            double du=0,dv=0;
            for (int i=0;i<4;++i) for (int j=0;j<4;++j) {
                const double ideal=k==0 ? i/3. : k==1 ? j/3. : 0.;
                error[k]=std::max(error[k],std::abs(row[k].Dot(net[i][j]-origin)-ideal));
                if (i<3) du=std::max(du,std::abs(3*row[k].Dot(net[i+1][j]-net[i][j])-(k==0)));
                if (j<3) dv=std::max(dv,std::abs(3*row[k].Dot(net[i][j+1]-net[i][j])-(k==1)));
            }
            gradient[k]=du+dv;
            pad[k]=8*tol*row[k].Modulus()+1.e-12;
        }
        regular=true;
    }
    enum class State { empty, interior, uncertain };
    double spatial_margin() const { return 8*tol; }
    struct Projection {
        std::array<double,3> point,speed;
    };
    Projection project(const MaterialRay& ray) const {
        const auto relative=ray.point-origin;
        return {{row[0].Dot(relative),row[1].Dot(relative),row[2].Dot(relative)},
                {row[0].Dot(ray.direction),row[1].Dot(ray.direction),row[2].Dot(ray.direction)}};
    }
    // A single interval encloses the depths of all possible finite crossings
    // in this packet. The projections use each material normal's true direction.
    State classify(std::span<const Projection> rays,double reach,double orientation) const {
        if (!regular) return State::uncertain;
        double low=0,high=reach+4*tol;
        std::array<ScalarRange,3> p,d;
        for (const int k:{2,0,1}) {
            const double x=rays.front().point[k],v=rays.front().speed[k];
            p[k]={x,x}; d[k]={v,v};
            for (const auto& ray:rays.subspan(1)) {
                p[k].include(ray.point[k]);
                d[k].include(ray.speed[k]);
            }
            const double margin=error[k]+pad[k], extra=4*tol*d[k].magnitude();
            // Negative near-origin roots are included with an enlarged slab.
            if (!clip(-margin-p[k].hi-extra,d[k].hi,low,high) ||
                !clip(p[k].lo-(k==2 ? 0. : 1.)-margin-extra,-d[k].lo,low,high)) return State::empty;
        }
        const double transverse=orientation>0 ? d[2].lo : -d[2].hi;
        if (!(transverse>.02)) return State::uncertain;
        // Use the supporting plane for existence. An interval clipped by a
        // side of the patch alone cannot establish an interior intersection.
        const auto times=plane_times(p[2],d[2]);
        const double spread=(error[2]+pad[2])/transverse;
        if (times.lo-spread<=8*tol || times.hi+spread>=reach-8*tol)
            return State::uncertain;
        for (int k=0;k<2;++k) {
            const double slope=d[k].magnitude()/transverse;
            const auto travel=product(times,d[k]);
            const double radius=error[k]+slope*error[2]+pad[k]+slope*pad[2];
            if (gradient[k]+slope*gradient[2]>=.8 ||
                p[k].lo+travel.lo-radius<=0 || p[k].hi+travel.hi+radius>=1) return State::uncertain;
        }
        return State::interior;
    }
    State classify(const Projection& ray,double reach,double orientation) const {
        if (!regular) return State::uncertain;
        double low=0.,high=reach+4*tol;
        for (const int k:{2,0,1}) {
            const double d=ray.speed[k],p=ray.point[k];
            const double width=error[k]+pad[k]+4*tol*std::abs(d);
            if (!clip(-width-p,d,low,high) || !clip(p-(k==2 ? 0. : 1.)-width,-d,low,high))
                return State::empty;
        }
        const double nd=ray.speed[2];
        if (!(orientation*nd>.02)) return State::uncertain;
        const double time=-ray.point[2]/nd,spread=(error[2]+pad[2])/std::abs(nd);
        if (time-spread<=8*tol || time+spread>=reach-8*tol) return State::uncertain;
        for (int k=0;k<2;++k) {
            const double slope=std::abs(ray.speed[k]/nd);
            const double radius=error[k]+pad[k]+slope*(error[2]+pad[2]);
            const double uv=ray.point[k]+time*ray.speed[k];
            if (uv-radius<=0 || uv+radius>=1 || gradient[k]+slope*gradient[2]>=.8)
                return State::uncertain;
        }
        return State::interior;
    }
private:
    double tol{};
    std::array<double,3> pad{};
    static bool clip(double need,double rate,double& low,double& high) {
        if (rate>0) low=std::max(low,need/rate);
        else if (rate<0) high=std::min(high,need/rate);
        else if (need>0) return false;
        return low<=high;
    }
    static ScalarRange product(ScalarRange a,ScalarRange b) {
        const auto terms={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
        return {std::min(terms),std::max(terms)};
    }
    static ScalarRange plane_times(ScalarRange p,ScalarRange d) {
        return product({-p.hi,-p.lo},{1/d.hi,1/d.lo});
    }
};
}
