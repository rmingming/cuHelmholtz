#ifndef __FUNC_INTERFACE__
#define __FUNC_INTERFACE__
#include "global.h"

void do_transform(double *d_data,
                    double *d_out,
                    int NX,
                    int NY,
                    int NZ,
                    transform_kind X_TRANS,
                    transform_kind Y_TRANS,
                    transform_kind Z_TRANS);

void do_transform_nocubic(double *d_data,
                    double *d_out,
                    int NX,
                    int NY,
                    int NZ,
                    transform_kind X_TRANS,
                    transform_kind Y_TRANS,
                    transform_kind Z_TRANS);

void do_transform_cubic(double *d_data,
                    int NX,
                    int NY,
                    int NZ,
                    transform_kind X_TRANS,
                    transform_kind Y_TRANS,
                    transform_kind Z_TRANS);


void freeMemory_cubic();
#endif /* __FUNC_INTERFACE__ */