// Frictionless normal contact with implicit linear compliance or hard unilateral
// constraints. One warp owns each coupled island. Its rows may use different laws.
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {
enum Law { penalty, multiplier, augmented };
__device__ double warp_max(double v) {
    for (int offset=16;offset;offset/=2) v=fmax(v,__shfl_down_sync(0xffffffffu,v,offset));
    return __shfl_sync(0xffffffffu,v,0);
}
__device__ double row_residual(double impulse,double speed,double diagonal) {
    if (!isfinite(speed) || !isfinite(impulse)) return CUDART_INF;
    // Projected-gradient residual expressed as a velocity, also valid when
    // an immovable, unloaded pair has zero diagonal mobility.
    return fabs(fmin(impulse*diagonal,speed));
}
// A two-row normal complementarity problem has four possible active sets.
// Solve nonsingular blocks in registers, including the inner proximal problem
// of each augmented-multiplier update. Singular blocks can use the general sweep.
__device__ bool solve_two(double a,double b,double c,double d,double u,double v,
                          double tolerance,double& x,double& y) {
    x=0.; y=0.;
    if (u>=-tolerance && v>=-tolerance) return true;
    if (a>0 && u<0 && v-c*u/a>=-tolerance) {x=-u/a;return true;}
    if (d>0 && v<0 && u-b*v/d>=-tolerance) {y=-v/d;return true;}
    const double det=a*d-b*c;
    if (!(det>1e-14*(fabs(a*d)+fabs(b*c)))) return false;
    x=(b*v-d*u)/det; y=(c*u-a*v)/det;
    return x>=0 && y>=0 && isfinite(x) && isfinite(y);
}
}

extern "C" __global__ void solve_normal_rows(const double* mobility,const double* free_gap,
        const int* laws,const double* stiffness,int rows,double h,double tolerance,
        int max_sweeps,double* impulse,double* residual,int* sweeps,int* status) {
    const int island=blockIdx.x,lane=threadIdx.x;
    const unsigned mask=0xffffffffu;
    const size_t first=(size_t)island*rows;
    const double* w=mobility+first*rows;
    extern __shared__ double workspace[];
    double* speed=workspace;
    double* p=speed+rows;
    double* anchor=p+rows;
    double* compliance=anchor+rows;
    double* proximal=compliance+rows;
    bool bad=false,has_augmented=false;
    for (int i=lane;i<rows;i+=32) {
        const int law=laws[first+i];
        const double diag=w[i*rows+i];
        bad|=law<penalty || law>augmented || diag<0 || !isfinite(diag) || !isfinite(free_gap[first+i]);
        if (law==penalty) bad|=!(stiffness[first+i]>0 && isfinite(stiffness[first+i]));
        compliance[i]=law==penalty ? 1./(stiffness[first+i]*h*h) : 0.;
        bad|=!isfinite(compliance[i]);
        proximal[i]=law==augmented ? .25*diag : 0.;
        has_augmented|=law==augmented;
        speed[i]=free_gap[first+i]/h;
        p[i]=0.;
        anchor[i]=0.;
    }
    for (int i=lane;i<rows*rows;i+=32) bad|=!isfinite(w[i]);
    bad=__any_sync(mask,bad);
    has_augmented=__any_sync(mask,has_augmented);
    __syncwarp();
    if (rows==2 && !bad) {
        int solved=0;
        if (lane==0) {
            const double a=w[0]+compliance[0],d=w[3]+compliance[1];
            const double b=w[1],c=w[2],u=speed[0],v=speed[1],r=proximal[0],s=proximal[1];
            double x=0.,y=0.;
            for (int it=0;it<max_sweeps;++it) {
                const double gu=u-r*x,gv=v-s*y;
                if (!solve_two(a+r,b,c,d+s,gu,gv,tolerance*.0625,x,y)) break;
                const double e=fmax(row_residual(x,u+a*x+b*y,a),row_residual(y,v+c*x+d*y,d));
                if (e<=tolerance || it+1==max_sweeps) {
                    impulse[first]=x;impulse[first+1]=y;
                    residual[island]=e;sweeps[island]=it+1;status[island]=e<=tolerance?0:1;
                    solved=1;break;
                }
                if (!has_augmented) break;
            }
        }
        if (__shfl_sync(mask,solved,0)) return;
    }
    int completed=0;
    double error=CUDART_INF;
    while (!bad && completed<max_sweeps) {
        for (int row=0;row<rows;++row) {
            double delta=0.;
            if (lane==0) {
                const double d=w[row*rows+row]+compliance[row]+proximal[row];
                if (d>0) {
                    const double v=speed[row]+compliance[row]*p[row]+proximal[row]*(p[row]-anchor[row]);
                    const double next=fmax(0.,p[row]-v/d);
                    delta=next-p[row];
                    p[row]=next;
                }
            }
            delta=__shfl_sync(mask,delta,0);
            for (int i=lane;i<rows;i+=32) speed[i]+=w[i*rows+row]*delta;
            __syncwarp();
        }
        ++completed;
        double physical=0.,inner=0.;
        for (int i=lane;i<rows;i+=32) {
            const double d=w[i*rows+i]+compliance[i];
            const double v=speed[i]+compliance[i]*p[i];
            physical=fmax(physical,row_residual(p[i],v,d));
            inner=fmax(inner,row_residual(p[i],v+proximal[i]*(p[i]-anchor[i]),d+proximal[i]));
        }
        error=warp_max(physical);
        if (error<=tolerance || !isfinite(error)) break;
        if (has_augmented && warp_max(inner)<=.125*tolerance) {
            for (int i=lane;i<rows;i+=32) anchor[i]=p[i];
            __syncwarp();
        }
    }
    for (int i=lane;i<rows;i+=32) impulse[first+i]=p[i];
    if (lane==0) {
        residual[island]=error;
        sweeps[island]=completed;
        status[island]=bad ? 2 : error<=tolerance ? 0 : 1;
    }
}
