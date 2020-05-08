#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define time_(...)                                                                       \
{			 									\
	struct timeval __pre, __now;									\
	gettimeofday(&__pre,0);									\
	__VA_ARGS__										\
	gettimeofday(&__now,0);									\
	printf("||||||||||||||||||| %s:%d: \t%.6f\n",  __FILE__, __LINE__,						\
		__now.tv_sec + __now.tv_usec/1000000.0 - __pre.tv_sec - __pre.tv_usec/1000000.0);	\
	fflush(NULL);										\
}

