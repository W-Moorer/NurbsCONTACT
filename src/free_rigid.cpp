#include "nurbscontact/free_rigid.hpp"

namespace nc::dynamics {
namespace {
Quaternion read4(const Vector& v,std::size_t i) { return {v.at(i),v.at(i+1),v.at(i+2),v.at(i+3)}; }
Vec imag(Quaternion a) { return {a[1],a[2],a[3]}; }
Quaternion quat(double real,Vec v) { return {real,v[0],v[1],v[2]}; }
Quaternion multiply(Quaternion a,Quaternion b) {
    const Vec u=imag(a),w=imag(b);
    return quat(a[0]*b[0]-dot3(u,w),add(add(scale(w,a[0]),scale(u,b[0])),cross3(u,w)));
}
double norm2(Quaternion a) { return a[0]*a[0]+squared(imag(a)); }
Quaternion inverse(Quaternion a) {
    const double n=norm2(a);
    if (!(std::isfinite(n) && n>1e-20)) throw std::runtime_error("singular free-body quaternion");
    return quat(a[0]/n,scale(imag(a),-1/n));
}
Vec matvec(const std::array<Vec,3>& m,Vec x) { return {dot3(m[0],x),dot3(m[1],x),dot3(m[2],x)}; }
}

FreeRigidBody::FreeRigidBody(std::size_t start,Vec d,Vec p)
    : first(start),inertia{{{d[0],p[0],p[1]},{p[0],d[1],p[2]},{p[1],p[2],d[2]}}} {
    const auto c0=cross3(inertia[1],inertia[2]),c1=cross3(inertia[2],inertia[0]),c2=cross3(inertia[0],inertia[1]);
    const double det=dot3(inertia[0],c0);
    if (!(d[0]>0 && d[0]*d[1]>p[0]*p[0] && det>0 && std::isfinite(det)))
        throw std::invalid_argument("free-body inertia tensor must be symmetric positive definite");
    inverse_inertia={scale(c0,1/det),scale(c1,1/det),scale(c2,1/det)};
}

Quaternion FreeRigidBody::initial_attitude(const ContactPose& pose) const {
    return quat(std::cos(pose.angle*.5),scale(pose.axis,std::sin(pose.angle*.5)));
}
Quaternion FreeRigidBody::initial_rate(const Quaternion& q,Vec omega) const {
    return multiply(q,quat(0,scale(omega,.5)));
}

void FreeRigidBody::angular_impulse_response(const Vector& q,const Vector& generalized,Vector& dv) const {
    const auto i=first+3;
    const auto rot=read4(q,i);
    const auto moment=scale(imag(multiply(inverse(rot),read4(generalized,i))),.5*norm2(rot));
    const auto rate=initial_rate(rot,matvec(inverse_inertia,moment));
    for (std::size_t j=0;j<4;++j) dv[i+j]=rate[j];
}

Vec FreeRigidBody::body_angular_velocity(const Vector& q,const Vector& v) const {
    return scale(imag(multiply(inverse(read4(q,first+3)),read4(v,first+3))),2.);
}
Vec FreeRigidBody::body_angular_acceleration(const State& state) const {
    const auto inv=inverse(read4(state.q,first+3));
    const auto w=multiply(inv,read4(state.v,first+3)),acc=multiply(inv,read4(state.a,first+3));
    return scale(sub(imag(acc),imag(multiply(w,w))),2.);
}

void FreeRigidBody::kinematics(const Vector& q,Vec local_com,ContactBodyState& state) const {
    const Quaternion rot=read4(q,first+3), inv=inverse(rot);
    const double sine=length(imag(rot));
    state.pose.axis=sine>0 ? scale(imag(rot),1/sine) : Vec{0,0,1};
    state.pose.angle=2*std::atan2(sine,rot[0]);
    const Vec from_com=scale(state.pose.rotate(local_com),-1.);
    state.pose.origin=add({q[first],q[first+1],q[first+2]},from_com);
    for (std::size_t i=0;i<3;++i) state.derivative[first+i].linear[i]=1.;
    for (std::size_t i=0;i<4;++i) {
        Quaternion e{};
        e[i]=1.;
        const Vec world_axis=scale(imag(multiply(e,inv)),2.);
        state.derivative[first+3+i]={cross3(world_axis,from_com),world_axis};
    }
}

void FreeRigidBody::apply_euler_equations(const Vector& q,const Vector& v,Forces& forces) const {
    const std::size_t i=first+3,n=q.size();
    const auto rot=read4(q,i),inv=inverse(rot),rate=multiply(inv,read4(v,i));
    const Vec omega=scale(imag(rate),2.);
    // For the quaternion Jacobian B, B B^T = 4 / |q|² I. Recover the
    // body torque from B^T tau (the generalized contact and load forces).
    const auto torque=[&](const Quaternion& generalized) {
        return scale(imag(multiply(inv,generalized)),.5*norm2(rot));
    };
    const auto momentum=matvec(inertia,omega);
    const Vec alpha=matvec(inverse_inertia,sub(torque(read4(forces.value,i)),cross3(omega,momentum)));
    const auto acceleration=multiply(rot,quat(-.25*squared(omega),add(scale(alpha,.5),scale(omega,rate[0]))));
    for (std::size_t k=0;k<4;++k) forces.value[i+k]=acceleration[k];
    // Transform the existing contact tangents into the same acceleration
    // equations. The Newton residual always includes the full gyroscopic term.
    for (Vector* tangent : {&forces.stiffness,&forces.damping}) {
        for (std::size_t j=0;j<n;++j) {
            Quaternion column;
            for (std::size_t k=0;k<4;++k) column[k]=(*tangent)[(i+k)*n+j];
            const auto mapped=initial_rate(rot,matvec(inverse_inertia,torque(column)));
            for (std::size_t k=0;k<4;++k) (*tangent)[(i+k)*n+j]=mapped[k];
        }
    }
}

void FreeRigidBody::normalize(State& state) const {
    const auto rot=read4(state.q,first+3);
    const double norm=std::sqrt(norm2(rot));
    if (!(norm>1e-10 && std::isfinite(norm))) throw std::runtime_error("invalid accepted free-body attitude");
    // A common rescaling leaves attitude, omega, alpha and the HHT force
    // history invariant in physical units.
    for (Vector* x : {&state.q,&state.v,&state.a,&state.force})
        for (std::size_t k=first+3;k<first+7;++k) (*x)[k]/=norm;
}
} // namespace nc::dynamics
