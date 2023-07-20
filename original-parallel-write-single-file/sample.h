#ifdef SAMPLE_PRE_LOOP
// START SAMPLE_PRE_LOOP

#ifndef __NVCC__
extern float* ch1dxx;
extern float* ch1dyy;
extern float* ch1dzz;
extern float* ch1dxy;
extern float* ch1dyz;
extern float* ch1dxz;
extern float* v2px;
extern float* v2pz;
extern float* v2sz;
extern float* v2pn;
#endif

const int strideX=ind(1,0,0)-ind(0,0,0);
const int strideY=ind(0,1,0)-ind(0,0,0);
const int strideZ=ind(0,0,1)-ind(0,0,0);

const float dxxinv=1.0f/(dx*dx);
const float dyyinv=1.0f/(dy*dy);
const float dzzinv=1.0f/(dz*dz);
const float dxyinv=1.0f/(dx*dy);
const float dxzinv=1.0f/(dx*dz);
const float dyzinv=1.0f/(dy*dz);

// END SAMPLE_PRE_LOOP
#endif


#ifdef SAMPLE_LOOP
// START ONE SAMPLE

const int i=ind(ix,iy,iz);

// p derivatives, H1(p) and H2(p)

const float pxx= Der2(pc, i, strideX, dxxinv);
const float pyy= Der2(pc, i, strideY, dyyinv);
const float pzz= Der2(pc, i, strideZ, dzzinv);
const float pxy= DerCross(pc, i, strideX, strideY, dxyinv);
const float pyz= DerCross(pc, i, strideY, strideZ, dyzinv);
const float pxz= DerCross(pc, i, strideX, strideZ, dxzinv);

const float cpxx=ch1dxx[i]*pxx;
const float cpyy=ch1dyy[i]*pyy;
const float cpzz=ch1dzz[i]*pzz;
const float cpxy=ch1dxy[i]*pxy;
const float cpxz=ch1dxz[i]*pxz;
const float cpyz=ch1dyz[i]*pyz;
const float h1p=cpxx+cpyy+cpzz+cpxy+cpxz+cpyz;
const float h2p=pxx+pyy+pzz-h1p;

// q derivatives, H1(q) and H2(q)

const float qxx= Der2(qc, i, strideX, dxxinv);
const float qyy= Der2(qc, i, strideY, dyyinv);
const float qzz= Der2(qc, i, strideZ, dzzinv);
const float qxy= DerCross(qc, i, strideX,  strideY, dxyinv);
const float qyz= DerCross(qc, i, strideY,  strideZ, dyzinv);
const float qxz= DerCross(qc, i, strideX,  strideZ, dxzinv);

const float cqxx=ch1dxx[i]*qxx;
const float cqyy=ch1dyy[i]*qyy;
const float cqzz=ch1dzz[i]*qzz;
const float cqxy=ch1dxy[i]*qxy;
const float cqxz=ch1dxz[i]*qxz;
const float cqyz=ch1dyz[i]*qyz;
const float h1q=cqxx+cqyy+cqzz+cqxy+cqxz+cqyz;
const float h2q=qxx+qyy+qzz-h1q;

// p-q derivatives, H1(p-q) and H2(p-q)

const float h1pmq=h1p-h1q;
const float h2pmq=h2p-h2q;

// rhs of p and q equations

const float rhsp=v2px[i]*h2p + v2pz[i]*h1q + v2sz[i]*h1pmq;
const float rhsq=v2pn[i]*h2p + v2pz[i]*h1q - v2sz[i]*h2pmq;

// new p and q

pp[i]=2.0f*pc[i] - pp[i] + rhsp*dt*dt;
qp[i]=2.0f*qc[i] - qp[i] + rhsq*dt*dt;

// END ONE SAMPLE
#endif

#ifdef SAMPLE_POST_LOOP
#endif

