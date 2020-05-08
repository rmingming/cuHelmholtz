#ifndef __TRANSFORM_FUNCS__
#define __TRANSFORM_FUNCS__
#include <cufft.h>
#include <cufftXt.h>
#include "global.h"

/*
* 3D dst-1 transform for x dim
 */
void run_3d_dst_1_inplace_nocubic(double *in , int DATA_SIZE,int batch,int nLayer);
void freeMemory_dst1();



/*
* 3D dst-3 transform for x dim
 */
void run_3d_dst_3_inplace_nocubic(double *d_data , int DATA_SIZE,int batch,int nLayer);
void freeMemory_dst3();



/*
* 3D dct-1 transform for x dim
 */
void run_3d_dct_1_inplace_nocubic(double *d_data , int DATA_SIZE,int batch,int nLayer);
void freeMemory_dct1_nocubic();



/*
* 3D dct-2 transform for x dim
 */
void run_3d_dct_2_inplace_nocubic(double *d_data , int DATA_SIZE,int batch ,int nLayer);
void freeMemory_dct2();



/*
* 3D dct-3 transform for x dim
 */
void run_3d_dct_3_inplace_nocubic(double *d_data , int DATA_SIZE,int batch,int nLayer);
void freeMemory_dct3();


/*
* 3D dst-2 transform for x dim
 */
void run_3d_dst_2_inplace_nocubic(double *d_data , int DATA_SIZE, int batch ,int nLayer);
void freeMemory_dst2();





//inplace


/*
* 3D dst-1 transform for x dim
 */
void run_3d_dst_1_inplace(double *in , int DATA_SIZE,cufftHandle &plan_dst1);




/*
* 3D dst-3 transform for x dim
 */
void run_3d_dst_3_inplace(double *d_data , int DATA_SIZE,cufftHandle &plan_dst3);




/*
* 3D dct-1 transform for x dim
 */
void run_3d_dct_1_inplace(double *d_data , int DATA_SIZE,cufftHandle &plan_dct1);
void freeMemory_dct1_cubic();



/*
* 3D dct-2 transform for x dim
 */
void run_3d_dct_2_inplace(double *d_data , int DATA_SIZE,cufftHandle &plan_dct2);




/*
* 3D dct-3 transform for x dim
 */
void run_3d_dct_3_inplace(double *d_data , int DATA_SIZE,cufftHandle &plan_dct3);



/*
* 3D dst-2 transform for x dim
 */
void run_3d_dst_2_inplace(double *d_data , int DATA_SIZE,cufftHandle &plan_dst2);



void run_dft_r2c_inplace(double *d_data , int DATA_SIZE,cufftHandle &plan_dft_r2c_cubic);

void run_dft_c2r_inplace(double *d_data , int DATA_SIZE,cufftHandle &plan_dft_c2r_cubic);

void run_dft_r2c_inplace_nocubic(double *d_data , int DATA_SIZE,int batch,int nLayer);

void run_dft_c2r_inplace_nocubic(double *d_data , int DATA_SIZE,int batch,int nLayer);

void freeMemory_r2c();

void freeMemory_c2r();



#endif /* __TRANSFORM_FUNCS__ */
