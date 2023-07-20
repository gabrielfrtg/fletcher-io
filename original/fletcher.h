#define MI 0.2           // stability factor to compute dt
#define ARGS 11          // tokens in executable command

#define _DUMP       // execution summary dump
//#undef  _DUMP     // execution summary dump

//#define SIGMA 20.0     // value of sigma (infinity) on formula 7 of Fletcher's paper
//#define SIGMA  6.0     // value of sigma on formula 7 of Fletcher's paper
//#define SIGMA  1.5     // value of sigma on formula 7 of Fletcher's paper
#define SIGMA  0.75      // value of sigma on formula 7 of Fletcher's paper
#define MAX_SIGMA 10.0   // above this value, SIGMA is considered infinite; as so, vsz=0


