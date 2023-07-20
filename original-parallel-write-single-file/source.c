#include "source.h"


// Source: compute source value at time it*dt


float Source(float dt, int it){

  float tf, fc, fct, expo;
  tf=TWOSQRTPI/FCUT;
  fc=FCUT/THREESQRTPI;
  fct=fc*(((float)it)*dt-tf);
  expo=PICUBE*fct*fct;
  return ((1.0f-2.0f*expo)*expf(-expo));
}
