#include "nurbscontact/surface_impulse.hpp"

#include <algorithm>
#include <numbers>
#include <numeric>
#include <string>

namespace nc::dynamics {
namespace {
// Contact coordinates are tangent 1, tangent 2, normal.
struct Block {
    std::size_t pair;
    ContactRegion region;
    std::array<Vec,3> axes;
    std::array<Vector,3> jacobian, response;
    Vec bias{}, reg{}, metric{}, impulse{};
    std::array<Vec,3> mobility{};
    double mu{}, free_normal_speed{}, impact_speed{};
};

double product(const Vector& a,const Vector& b) {
    return std::inner_product(a.begin(),a.end(),b.begin(),0.);
}

std::array<Vec,3> frame(Vec n) {
    // Pick the coordinate axis least parallel to the surface normal.
    std::size_t k=0;
    for (std::size_t i=1;i<3;++i) if (std::abs(n[i])<std::abs(n[k])) k=i;
    Vec axis{};
    axis[k]=1;
    const auto t=cross3(n,axis);
    const auto u=scale(t,1./length(t));
    return {u,cross3(n,u),n};
}

Vec project_cone(Vec y,Vec metric,double mu) {
    const double r=std::hypot(y[0],y[1]);
    if (r<=mu*y[2] && y[2]>=0) return y;
    const double normal=(metric[2]*y[2]+mu*metric[0]*r)/(metric[2]+mu*mu*metric[0]);
    if (normal<=0) return {};
    const double s=r>0 ? mu*normal/r : 0.;
    return {s*y[0],s*y[1],normal};
}

// Four-generator friction pyramid. The tangential maximum-dissipation LP is
// equivalently characterized by projection onto |p1| + |p2| <= mu * pn.
Vec project_pyramid(Vec y,double mu) {
    const double n=std::max(0.,y[2]),radius=mu*n;
    const double a=std::abs(y[0]),b=std::abs(y[1]);
    const double threshold=std::max({0.,(a+b-radius)*.5,std::max(a,b)-radius});
    return {std::copysign(std::max(0.,a-threshold),y[0]),
            std::copysign(std::max(0.,b-threshold),y[1]),n};
}

// Solve the local unilateral normal multiplier and the tangential
// maximum-dissipation problem in its physical 2D mass metric.
Vec multiplier_step(const Block& b,Vec gradient) {
    const auto& w=b.mobility;
    const double n=std::max(0.,b.impulse[2]-gradient[2]/w[2][2]);
    const double radius=b.mu*n;
    if (radius==0) return {0,0,n};
    Vec rhs{};
    for (int k=0;k<2;++k)
        rhs[k]=w[k][0]*b.impulse[0]+w[k][1]*b.impulse[1]-gradient[k]-w[k][2]*(n-b.impulse[2]);
    // The disk multiplier shifts both eigenvalues of the tangential metric.
    const double angle=.5*std::atan2(2*w[0][1],w[0][0]-w[1][1]);
    const double c=std::cos(angle),s=std::sin(angle);
    const double p=c*rhs[0]+s*rhs[1],q=-s*rhs[0]+c*rhs[1];
    const double center=.5*(w[0][0]+w[1][1]),spread=.5*std::hypot(w[0][0]-w[1][1],2*w[0][1]);
    const double a=center+spread,d=std::max(0.,center-spread);
    const auto component=[](double v,double diag) {
        return diag>0 ? v/diag : v==0 ? 0. : std::copysign(std::numeric_limits<double>::infinity(),v);
    };
    double x=component(p,a),y=component(q,d);
    if (std::hypot(x,y)>radius) {
        double lo=0.,hi=std::hypot(p,q)/radius;
        for (int i=0;i<60;++i) {
            const double mid=std::midpoint(lo,hi);
            if (std::hypot(component(p,a+mid),component(q,d+mid))>radius) lo=mid;
            else hi=mid;
        }
        x=component(p,a+hi); y=component(q,d+hi);
    }
    return {c*x-s*y,s*x+c*y,n};
}

Vec proposal(const Block& block,const Vector& dv,ImpulseLaw law) {
    Vec y,gradient;
    for (int i=0;i<3;++i) {
        gradient[i]=block.bias[i]+product(block.jacobian[i],dv)+block.reg[i]*block.impulse[i];
        y[i]=block.impulse[i]-gradient[i]/block.metric[i];
    }
    if (law==ImpulseLaw::sap) return project_cone(y,block.metric,block.mu);
    if (law==ImpulseLaw::lcp) return project_pyramid(y,block.mu);
    if (law==ImpulseLaw::lagrange) return multiplier_step(block,gradient);
    if (law==ImpulseLaw::soccp) {
        // De Saxce's modified normal velocity gives the non-associated
        // Coulomb law as complementarity with the dual of the friction cone.
        y[2]-=block.mu*std::hypot(gradient[0],gradient[1])/block.metric[2];
        return project_cone(y,block.metric,block.mu);
    }
    // Signorini min-map and maximum dissipation on the circular disk.
    const double n=std::max(0.,y[2]);
    const double r=std::hypot(y[0],y[1]);
    const double s=r>0 ? std::min(1.,block.mu*n/r) : 1.;
    return {s*y[0],s*y[1],n};
}

double natural_residual(const std::vector<Block>& blocks,const Vector& dv,ImpulseLaw law) {
    double residual=0.;
    for (const auto& block:blocks) {
        if (!block.metric[0]) continue;
        // Check the underlying disk KKT conditions for the multiplier methods.
        const auto check=law==ImpulseLaw::lagrange ? ImpulseLaw::ncp : law;
        const auto diff=sub(proposal(block,dv,check),block.impulse);
        for (int k=0;k<3;++k) {
            const double error=std::abs(diff[k])*block.metric[k];
            if (!std::isfinite(error)) throw UnresolvedContact("nonfinite impulse residual");
            residual=std::max(residual,error);
        }
    }
    return residual;
}

void solve_blocks(std::vector<Block>& blocks,SurfaceImpulseResult& result,ImpulseLaw law,double tol) {
    auto& dv=result.velocity_change;
    for (unsigned sweep=0;sweep<4000;++sweep) {
        for (auto& block:blocks) {
            if (!block.metric[0]) continue;
            const Vec proposed=proposal(block,dv,law);
            for (int k=0;k<3;++k)
                for (std::size_t j=0;j<dv.size();++j)
                    dv[j]+=block.response[k][j]*(proposed[k]-block.impulse[k]);
            block.impulse=proposed;
        }
        ++result.iterations;
        result.residual_m_s=natural_residual(blocks,dv,law);
        if (result.residual_m_s<=tol) return;
    }
    throw UnresolvedContact("contact impulse iteration did not converge");
}

void augmented_solve(std::vector<Block>& blocks,SurfaceImpulseResult& result,double tol) {
    // A proximal dual solve is the unilateral normal augmented-Lagrangian
    // update p_next = max(0, p_old - rho*w_next). Use the disk update for
    // friction. Remove the proximal term before checking physical residuals.
    std::vector<Vec> original;
    for (const auto& b:blocks) original.push_back(b.bias);
    for (unsigned outer=0;outer<500;++outer) {
        for (std::size_t i=0;i<blocks.size();++i) {
            auto& b=blocks[i];
            const double inverse_rho=.25*b.metric[0];
            b.reg.fill(inverse_rho);
            b.metric=add(b.metric,b.reg);
            b.bias=sub(original[i],scale(b.impulse,inverse_rho));
        }
        solve_blocks(blocks,result,ImpulseLaw::ncp,.125*tol);
        for (std::size_t i=0;i<blocks.size();++i) {
            auto& b=blocks[i];
            b.metric=sub(b.metric,b.reg);
            b.reg={};
            b.bias=original[i];
        }
        result.residual_m_s=natural_residual(blocks,result.velocity_change,ImpulseLaw::ncp);
        if (result.residual_m_s<=tol) return;
    }
    throw UnresolvedContact("augmented contact multiplier iteration did not converge");
}
}

std::string_view impulse_law_name(ImpulseLaw law) {
    switch (law) {
    case ImpulseLaw::sap: return "sap";
    case ImpulseLaw::lcp: return "lcp";
    case ImpulseLaw::ncp: return "ncp";
    case ImpulseLaw::soccp: return "soccp";
    case ImpulseLaw::lagrange: return "lagrange";
    case ImpulseLaw::augmented_lagrange: return "augmented-lagrange";
    }
    throw std::invalid_argument("unknown impulse law");
}
ImpulseLaw parse_impulse_law(std::string_view name) {
    for (const auto law:{ImpulseLaw::sap,ImpulseLaw::lcp,ImpulseLaw::ncp,ImpulseLaw::soccp,
                         ImpulseLaw::lagrange,ImpulseLaw::augmented_lagrange})
        if (name==impulse_law_name(law)) return law;
    throw std::invalid_argument("unknown surface contact solver");
}

SurfaceImpulseResult surface_impulses(ImpulseLaw law,std::span<const ContactBodyState> bodies,
        std::span<const ContactPair> pairs,const Vector& velocity,double h,
        const InverseMass& inverse_mass,ContactWork& work,std::span<const ContactSample> previous) {
    if (!(std::isfinite(h) && h>0)) throw std::invalid_argument("invalid impulse step duration");
    if (!previous.empty() && previous.size()!=pairs.size()) throw std::invalid_argument("invalid impact history");
    SurfaceImpulseResult result{Vector(velocity.size()),std::vector<ContactSample>(pairs.size())};
    std::vector<Block> blocks;
    double tolerance=1e-9;
    for (std::size_t i=0;i<pairs.size();++i) {
        const auto& pair=pairs[i];
        if (!pair.regions) throw std::invalid_argument("impulse solver requires a surface contact method");
        ++work.pairs;
        const auto& a=bodies[pair.a];
        const auto& b=bodies[pair.b];
        const auto regions=pair.regions->regions(a.pose,b.pose,work);
        if (regions.empty()) ++work.culled;
        if (law==ImpulseLaw::sap && regions.size()>1 && pair.material.approach_floor>0)
            throw std::runtime_error("incidence-dependent SAP damping needs per-region impact histories");
        for (const auto& region:regions) {
            if (!(region.depth_m>0 && region.depth_m<=pair.material.maximum_depth))
                throw UnresolvedContact("predicted overlap "+std::to_string(region.depth_m*1e6)+
                    " um outside the range (0, "+std::to_string(pair.material.maximum_depth*1e6)+
                    "] um for contact pair "+std::to_string(i));
            Block block{};
            block.pair=i;
            block.region=region;
            block.axes=frame(region.normal);
            block.mu=pair.material.friction;
            const Vec ra=sub(region.point,a.pose.origin),rb=sub(region.point,b.pose.origin);
            const auto prescribed=sub(b.prescribed.at(rb),a.prescribed.at(ra));
            Vec diagonal{};
            for (int k=0;k<3;++k) {
                auto& row=block.jacobian[k];
                for (std::size_t j=0;j<velocity.size();++j)
                    row.push_back(dot3(block.axes[k],sub(b.derivative[j].at(rb),a.derivative[j].at(ra))));
                block.response[k]=inverse_mass(row);
                if (block.response[k].size()!=velocity.size()) throw std::invalid_argument("inverse mass dimension mismatch");
                diagonal[k]=product(row,block.response[k]);
                block.bias[k]=dot3(block.axes[k],prescribed)+product(row,velocity);
            }
            block.free_normal_speed=block.bias[2];
            if (law==ImpulseLaw::lagrange)
                for (int k=0;k<3;++k) for (int j=0;j<3;++j)
                    block.mobility[k][j]=product(block.jacobian[k],block.response[j]);
            // A block Lipschitz bound for J M^-1 J^T. Each block uses its
            // latest neighbors' velocities during the projected sweep.
            const double bound=diagonal[0]+diagonal[1]+diagonal[2];
            if (law==ImpulseLaw::sap) {
                const auto& m=pair.material;
                const double power=std::pow(region.depth_m,m.exponent-1);
                const double k=m.exponent*m.stiffness*power;
                double tau=m.rate*std::sqrt(power)/k;
                if (m.approach_floor>0) {
                    const bool held=!previous.empty() && previous[i].gap<0 && previous[i].impact_speed>0;
                    block.impact_speed=held ? previous[i].impact_speed : std::max(m.approach_floor,-block.free_normal_speed);
                    tau=region.depth_m*m.impact_dissipation/(m.exponent*block.impact_speed);
                }
                const double rms=length(diagonal)/std::sqrt(3.);
                const double rt=1e-3*std::max(.5*(diagonal[0]+diagonal[1]),rms);
                const double rn=std::max(std::max(diagonal[2],rms)/(4*std::numbers::pi*std::numbers::pi),
                                         1./(h*k*(h+tau)));
                block.reg={rt,rt,rn};
                // Linearize k0*d^p at the predicted depth. Its elastic
                // load divided by tangent stiffness is d/p.
                block.bias[2]=(tau*block.free_normal_speed-region.depth_m/m.exponent)/(h+tau);
            } else {
                block.bias[2]=-region.depth_m/h;
                if (!(diagonal[2]>0)) throw UnresolvedContact("prescribed penetration has no dynamic normal mobility");
            }
            for (int k=0;k<3;++k) block.metric[k]=bound+block.reg[k];
            if (bound==0 && law==ImpulseLaw::sap) {
                // With no dynamic mobility the regularized cone has an
                // explicit solution in the zero tangential regularization limit.
                const double speed=std::hypot(block.bias[0],block.bias[1]);
                const double gn=std::max(0.,(block.mu*speed-block.bias[2])/block.reg[2]);
                const double factor=speed>0 ? -block.mu*gn/speed : 0.;
                block.impulse={factor*block.bias[0],factor*block.bias[1],gn};
            }
            if (!std::isfinite(length(block.bias)) || !std::isfinite(length(block.metric)))
                throw UnresolvedContact("nonfinite contact coefficients");
            tolerance=std::max(tolerance,1e-9*(1+length(block.bias)));
            blocks.push_back(std::move(block));
        }
    }
    auto& dv=result.velocity_change;
    result.regions=blocks.size();
    if (!blocks.empty()) {
        if (law==ImpulseLaw::augmented_lagrange) augmented_solve(blocks,result,tolerance);
        else solve_blocks(blocks,result,law,tolerance);
    }
    for (const auto& block:blocks) {
        auto& contact=result.contacts[block.pair];
        const double change=product(block.jacobian[2],dv);
        const double gap=-block.region.depth_m+h*change;
        if (!contact.resolved || gap<contact.gap) {
            contact.gap=gap;
            contact.gap_rate=block.free_normal_speed+change;
            contact.impact_speed=block.impact_speed;
            contact.witness_a=contact.witness_b=block.region.point;
        }
        contact.resolved=true;
        contact.normal_force+=block.impulse[2]/h;
        for (int k=0;k<2;++k) contact.friction=add(contact.friction,scale(block.axes[k],block.impulse[k]/h));
    }
    return result;
}
} // namespace nc::dynamics
