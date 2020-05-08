#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include "device_functions.h"

#include "transformfunc.h"
#include "funcinterface.h"
#include "cutranspose.h"
#include "time_.h"


// cufftHandle plan;
cufftHandle plan;
cufftHandle plan_c2r;
int do_r2c,do_c2r;

//会改变输入数组
void do_transform(double *d_data,double *d_out,int NX,int NY,int NZ,transform_kind X_TRANS,transform_kind Y_TRANS,transform_kind Z_TRANS){
    if(d_out==NULL||d_out==d_data){
        if(NX==NY&&NX==NZ){
            // printf("do inplace\n");
            do_transform_cubic(d_data,NX,NY,NZ,X_TRANS,Y_TRANS,Z_TRANS);
        }else{
            printf("the matrix for inplace transform must be cubic!\n");
            exit(0);
        }
    }else{
        if(d_out!=d_data){
            // printf("do outoplace\n");
            do_transform_nocubic(d_data,d_out,NX,NY,NZ,X_TRANS,Y_TRANS,Z_TRANS);
        }
    }
}

void do_transform_cubic(double *d_data,int NX,int NY,int NZ,transform_kind X_TRANS,transform_kind Y_TRANS,transform_kind Z_TRANS){
  
    do_c2r=0;
    do_r2c=0;
    
    
    if(X_TRANS==DST_2||Y_TRANS==DST_2||Z_TRANS==DST_2||
        X_TRANS==DFT_C2R||Y_TRANS==DFT_C2R||Z_TRANS==DFT_C2R||
        X_TRANS==DCT_2||Y_TRANS==DCT_2||Z_TRANS==DCT_2||
        X_TRANS==DST_1||Y_TRANS==DST_1||Z_TRANS==DST_1||
        X_TRANS==DCT_1||Y_TRANS==DCT_1||Z_TRANS==DCT_1){
        do_c2r=1;
        int n[1]={NX};
        int inembeb[1]={(NX+2)/2};
        int onembeb[1]={(NX+2)};
        cufftResult r = cufftPlanMany(&plan_c2r,1,n,
                    inembeb,1,(NX+2)/2,
                    onembeb,1,(NX+2),
                    CUFFT_Z2D, (NX+2)*(NX+2));

        if(r!=0){
            printf("CUFFT FAILED! ERROR CODE: %s\n",cufftresultcode[r]);
            exit(0);
        }
    }
    if(X_TRANS==DST_3||Y_TRANS==DST_3||Z_TRANS==DST_3||
        X_TRANS==DCT_3||Y_TRANS==DCT_3||Z_TRANS==DCT_3||
        X_TRANS==DFT_R2C||Y_TRANS==DFT_R2C||Z_TRANS==DFT_R2C){
        do_r2c=1;
        int n[1]={NX};
        int inembeb[1]={NX+2};
        int onembeb[1]={(NX+2)/2};
        cufftResult r = cufftPlanMany(&plan,1,n,
                    inembeb,1,NX+2,
                    onembeb,1,(NX+2)/2,
                    CUFFT_D2Z, (NX+2)*(NX+2));
        
        if(r!=0){
            printf("CUFFT FAILED! ERROR CODE: %s\n",cufftresultcode[r]);
            exit(0);
        }
    }
    
    
    if(X_TRANS==DST_1){
        // printf("x transfor dst1\n");
        run_3d_dst_1_inplace(d_data,NX,plan_c2r);
    }else if(X_TRANS==DST_2){
        run_3d_dst_2_inplace(d_data,NX,plan_c2r);
    }else if(X_TRANS==DST_3){
        // printf("x transfor dst3\n");
        run_3d_dst_3_inplace(d_data,NX+1,plan);
    }else if(X_TRANS==DCT_1){
        // printf("x transfor dct1\n");
        run_3d_dct_1_inplace(d_data,NX+1,plan_c2r);
    }else if(X_TRANS==DCT_2){
        // printf("x transfor dct2\n");
        run_3d_dct_2_inplace(d_data,NX,plan_c2r);
    }else if(X_TRANS==DCT_3){
        // printf("x transfor dct3\n");
        run_3d_dct_3_inplace(d_data,NX,plan);
    }else if(X_TRANS==DFT_R2C){
        run_dft_r2c_inplace(d_data,NX,plan);
    }else if(X_TRANS==DFT_C2R){
        run_dft_c2r_inplace(d_data,NX,plan_c2r);
    }else{
        printf("Please input the correct transform kind\n");
    }

	cudaDeviceSynchronize();
	cudaError_t e;
    if((e=cudaGetLastError())!=cudaSuccess){
        printf("CUDA ERROR: %s !\n",cudaGetErrorString(e));
        exit(0);
    }
    // printf("%s\n",cudaGetErrorString(cudaGetLastError()));

    
    int mat_size1[3]={NX+2,NY+2,NZ+2};
    int permutation1[3]={1,0,2};
    cut_transpose3d(d_data,d_data,mat_size1,permutation1,1);
	cudaDeviceSynchronize();



    if(Y_TRANS==DST_1){
        // printf("y transfor dst1\n");
        run_3d_dst_1_inplace(d_data,NY,plan_c2r);
    }else if(Y_TRANS==DST_2){
        run_3d_dst_2_inplace(d_data,NY,plan_c2r);
    }else if(Y_TRANS==DST_3){
        // printf("y transfor dst3\n");
        run_3d_dst_3_inplace(d_data,NY+1,plan);
    }else if(Y_TRANS==DCT_1){
        // printf("y transfor dct1\n");
        run_3d_dct_1_inplace(d_data,NY+1,plan_c2r);
    }else if(Y_TRANS==DCT_2){
        // printf("y transfor dct2\n");
        run_3d_dct_2_inplace(d_data,NY,plan_c2r);
    }else if(Y_TRANS==DCT_3){
        // printf("y transfor dct3\n");
        run_3d_dct_3_inplace(d_data,NY,plan);
    }else if(Y_TRANS==DFT_R2C){
        run_dft_r2c_inplace(d_data,NY,plan);
    }else if(Y_TRANS==DFT_C2R){
        run_dft_c2r_inplace(d_data,NY,plan_c2r);
    }else{
        printf("Please input the correct transform kind\n");
    }
	cudaDeviceSynchronize();
    if((e=cudaGetLastError())!=cudaSuccess){
        printf("CUDA ERROR: %s !\n",cudaGetErrorString(e));
        exit(0);
    }
    
    int mat_size2[3]={NY+2,NX+2,NZ+2};
    int permutation2[3]={2,0,1};
    cut_transpose3d(d_data,d_data,mat_size2,permutation2,1);
	cudaDeviceSynchronize();

    if(Z_TRANS==DST_1){
        // printf("z transfor dst1\n");
        run_3d_dst_1_inplace(d_data,NZ,plan_c2r);
    }else if(Z_TRANS==DST_2){
        run_3d_dst_2_inplace(d_data,NZ,plan_c2r);
    }else if(Z_TRANS==DST_3){
        // printf("z transfor dst3\n");
        run_3d_dst_3_inplace(d_data,NZ+1,plan);
    }else if(Z_TRANS==DCT_1){
        // printf("z transfor dct1\n");
        run_3d_dct_1_inplace(d_data,NZ+1,plan_c2r);
    }else if(Z_TRANS==DCT_2){
        // printf("z transfor dct2\n");
        run_3d_dct_2_inplace(d_data,NZ,plan_c2r);
    }else if(Z_TRANS==DCT_3){
        // printf("z transfor dct3\n");
        run_3d_dct_3_inplace(d_data,NZ,plan);
    }else if(Z_TRANS==DFT_R2C){
        run_dft_r2c_inplace(d_data,NZ,plan);
    }else if(Z_TRANS==DFT_C2R){
        run_dft_c2r_inplace(d_data,NZ,plan_c2r);
    }else{
        printf("Please input the correct transform kind\n");
    }
	cudaDeviceSynchronize();
    if((e=cudaGetLastError())!=cudaSuccess){
        printf("CUDA ERROR: %s !\n",cudaGetErrorString(e));
        exit(0);
    }
    
    int mat_size4[3]={NZ+2,NY+2,NX+2};
    int permutation4[3]={2,1,0};
    cut_transpose3d(d_data,d_data,mat_size4,permutation4,1);
	cudaDeviceSynchronize();

    freeMemory_cubic();
}

void do_transform_nocubic(double *d_data,double *d_out,int NX,int NY,int NZ,transform_kind X_TRANS,transform_kind Y_TRANS,transform_kind Z_TRANS){


    

    
    
    if(X_TRANS==DST_1){
        // printf("x transfor dst1\n");
        run_3d_dst_1_inplace_nocubic(d_data,NX,NY,NZ);
    }
    else if(X_TRANS==DST_2){
        run_3d_dst_2_inplace_nocubic(d_data,NX,NY,NZ);
    }
    else if(X_TRANS==DST_3){
        // printf("x transfor dst3\n");
        run_3d_dst_3_inplace_nocubic(d_data,NX+1,NY+1,NZ+1);
    }
    else if(X_TRANS==DCT_1){
        // printf("x transfor dct1\n");
        run_3d_dct_1_inplace_nocubic(d_data,NX+1,NY+1,NZ+1);
    }
    else if(X_TRANS==DCT_2){
        // printf("x transfor dct2\n");
        run_3d_dct_2_inplace_nocubic(d_data,NX,NY,NZ);
    }
    else if(X_TRANS==DCT_3){
        // printf("x transfor dct3\n");
        run_3d_dct_3_inplace_nocubic(d_data,NX,NY,NZ);
    }else if(X_TRANS==DFT_R2C){
        run_dft_r2c_inplace_nocubic(d_data,NX,NY,NZ);
    }else if(X_TRANS==DFT_C2R){
        run_dft_c2r_inplace_nocubic(d_data,NX,NY,NZ);
    }else{
        printf("Please input the correct transform kind\n");
    }
	cudaDeviceSynchronize();
    if(cudaGetLastError()!=cudaSuccess){
        printf("CUDA ERROR: %s !\n",cudaGetErrorString(cudaGetLastError()));
        exit(0);
    }

    
    int mat_size1[3]={NX+2,NY+2,NZ+2};
    int permutation1[3]={1,2,0};
    cut_transpose3d(d_out,d_data,mat_size1,permutation1,1);
	cudaDeviceSynchronize();



    if(Y_TRANS==DST_1){
        // printf("y transfor dst1\n");
        run_3d_dst_1_inplace_nocubic(d_out,NY,NZ,NX);
    }
    else if(Y_TRANS==DST_2){
        run_3d_dst_2_inplace_nocubic(d_out,NY,NZ,NX);
    }
    else if(Y_TRANS==DST_3){
        // printf("y transfor dst3\n");
        run_3d_dst_3_inplace_nocubic(d_out,NY+1,NZ+1,NX+1);
    }
    else if(Y_TRANS==DCT_1){
        // printf("y transfor dct1\n");
        run_3d_dct_1_inplace_nocubic(d_out,NY+1,NZ+1,NX+1);
    }
    else if(Y_TRANS==DCT_2){
        // printf("y transfor dct2\n");
        run_3d_dct_2_inplace_nocubic(d_out,NY,NZ,NX);
    }
    else if(Y_TRANS==DCT_3){
        // printf("y transfor dct3\n");
        run_3d_dct_3_inplace_nocubic(d_out,NY,NZ,NX);
    }else if(Y_TRANS==DFT_R2C){
        run_dft_r2c_inplace_nocubic(d_out,NY,NZ,NX);
    }else if(Y_TRANS==DFT_C2R){
        run_dft_c2r_inplace_nocubic(d_out,NY,NZ,NX);
    }else{
        printf("Please input the correct transform kind\n");
    }
	cudaDeviceSynchronize();
    if(cudaGetLastError()!=cudaSuccess){
        printf("CUDA ERROR: %s !\n",cudaGetErrorString(cudaGetLastError()));
        exit(0);
    }

    
    
    int mat_size2[3]={NY+2,NZ+2,NX+2};
    int permutation2[3]={1,2,0};
    cut_transpose3d(d_data,d_out,mat_size2,permutation2,1);
	cudaDeviceSynchronize();

    if(Z_TRANS==DST_1){
        // printf("z transfor dst1\n");
        run_3d_dst_1_inplace_nocubic(d_data,NZ,NX,NY);
    }
    else if(Z_TRANS==DST_2){
        run_3d_dst_2_inplace_nocubic(d_data,NZ,NX,NY);
    }
    else if(Z_TRANS==DST_3){
        // printf("z transfor dst3\n");
        run_3d_dst_3_inplace_nocubic(d_data,NZ+1,NX+1,NY+1);
    }
    else if(Z_TRANS==DCT_1){
        // printf("z transfor dct1\n");
        run_3d_dct_1_inplace_nocubic(d_data,NZ+1,NX+1,NY+1);
    }
    else if(Z_TRANS==DCT_2){
        // printf("z transfor dct2\n");
        run_3d_dct_2_inplace_nocubic(d_data,NZ,NX,NY);
    }
    else if(Z_TRANS==DCT_3){
        // printf("z transfor dct3\n");
        run_3d_dct_3_inplace_nocubic(d_data,NZ,NX,NY);
    }else if(Z_TRANS==DFT_R2C){
        run_dft_r2c_inplace_nocubic(d_data,NZ,NX,NY);
    }else if(Z_TRANS==DFT_C2R){
        run_dft_c2r_inplace_nocubic(d_data,NZ,NX,NY);
    }else{
        printf("Please input the correct transform kind\n");
    }
	cudaDeviceSynchronize();
    if(cudaGetLastError()!=cudaSuccess){
        printf("CUDA ERROR: %s !\n",cudaGetErrorString(cudaGetLastError()));
        exit(0);
    }
    // printf("%s\n",cudaGetErrorString(cudaGetLastError()));
    

    int mat_size3[3]={NZ+2,NX+2,NY+2};
    int permutation3[3]={1,2,0};
    cut_transpose3d(d_out,d_data,mat_size3,permutation3,1);
	cudaDeviceSynchronize();
    
}


void freeMemory_cubic(){
    if(do_r2c==1)
        cufftDestroy(plan);
    if(do_c2r==1)
        cufftDestroy(plan_c2r);
}




