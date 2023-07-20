#ifdef MODEL_GLOBALVARS
float *ch1dxx=NULL;  // isotropy simetry deep angle
float *ch1dyy=NULL;  // isotropy simetry deep angle
float *ch1dzz=NULL;  // isotropy simetry deep angle
float *ch1dxy=NULL;  // isotropy simetry deep angle
float *ch1dyz=NULL;  // isotropy simetry deep angle
float *ch1dxz=NULL;  // isotropy simetry deep angle
float *v2px=NULL;  // coeficient of H2(p)
float *v2pz=NULL;  // coeficient of H1(q)
float *v2sz=NULL;  // coeficient of H1(p-q) and H2(p-q)
float *v2pn=NULL;  // coeficient of H2(p)
#endif

#ifdef MODEL_INITIALIZE
// Precalcula campos abaixo

// coeficients of derivatives at H1 operator

ch1dxx = (float *) malloc(sx*sy*sz*sizeof(float));
ch1dyy = (float *) malloc(sx*sy*sz*sizeof(float));
ch1dzz = (float *) malloc(sx*sy*sz*sizeof(float));
ch1dxy = (float *) malloc(sx*sy*sz*sizeof(float));
ch1dyz = (float *) malloc(sx*sy*sz*sizeof(float));
ch1dxz = (float *) malloc(sx*sy*sz*sizeof(float));
for (int i=0; i<sx*sy*sz; i++) {
  float sinTheta=sin(theta[i]);
  float cosTheta=cos(theta[i]);
  float sin2Theta=sin(2.0*theta[i]);
  float sinPhi=sin(phi[i]);
  float cosPhi=cos(phi[i]);
  float sin2Phi=sin(2.0*phi[i]);
  ch1dxx[i]=sinTheta*sinTheta * cosPhi*cosPhi;
  ch1dyy[i]=sinTheta*sinTheta * sinPhi*sinPhi;
  ch1dzz[i]=cosTheta*cosTheta;
  ch1dxy[i]=sinTheta*sinTheta * sin2Phi;
  ch1dyz[i]=sin2Theta         * sinPhi;
  ch1dxz[i]=sin2Theta         * cosPhi;
}
#ifdef _DUMP
{
  const int iPrint=ind(bord+1,bord+1,bord+1);
  printf("ch1dxx=%f; ch1dyy=%f; ch1dzz=%f; ch1dxy=%f; ch1dxz=%f; ch1dyz=%f\n",
      ch1dxx[iPrint], ch1dyy[iPrint], ch1dzz[iPrint], ch1dxy[iPrint], ch1dxz[iPrint], ch1dyz[iPrint]);
}
#endif

// coeficients of H1 and H2 at PDEs

v2px = (float *) malloc(sx*sy*sz*sizeof(float));
v2pz = (float *) malloc(sx*sy*sz*sizeof(float));
v2sz = (float *) malloc(sx*sy*sz*sizeof(float));
v2pn = (float *) malloc(sx*sy*sz*sizeof(float));
for (int i=0; i<sx*sy*sz; i++){
  v2sz[i]=vsv[i]*vsv[i];
  v2pz[i]=vpz[i]*vpz[i];
  v2px[i]=v2pz[i]*(1.0+2.0*epsilon[i]);
  v2pn[i]=v2pz[i]*(1.0+2.0*delta[i]);
}
#ifdef _DUMP
{
  const int iPrint=ind(bord+1,bord+1,bord+1);
  printf("vsv=%e; vpz=%e, v2pz=%e\n",
         vsv[iPrint], vpz[iPrint], v2pz[iPrint]);
  printf("v2sz=%e; v2pz=%e, v2px=%e, v2pn=%e\n",
         v2sz[iPrint], v2pz[iPrint], v2px[iPrint], v2pn[iPrint]);
}
#endif

#endif

