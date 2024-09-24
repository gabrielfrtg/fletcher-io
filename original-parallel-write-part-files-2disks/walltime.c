#include <sys/time.h>

double wtime()
{
struct timeval tv;
struct timezone tz;
gettimeofday(&tv, &tz);
return tv.tv_sec + (1.0e-6*tv.tv_usec);
}
