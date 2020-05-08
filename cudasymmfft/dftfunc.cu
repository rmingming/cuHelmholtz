//in place 1d dst


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


# define M_PI   3.14159265358979323846


// #define TIME_TEST



cufftHandle plan_dft_r2c_nocubic;

cufftHandle plan_dft_c2r_nocubic;


#ifdef TIME_TEST
static long long time_preOp, time_postOp,time_trans_xzy,time_trans_zyx,time_cufft;


//static int conp_cnt=0;
static void timeBegin(struct timeval *tBegin){
	gettimeofday(tBegin, NULL);
}

static long long timeEnd(struct timeval tBegin){
    
	 struct timeval tEnd;
     gettimeofday(&tEnd, NULL);
    
     long long usec=(tEnd.tv_sec-tBegin.tv_sec)*1000*1000+tEnd.tv_usec-tBegin.tv_usec;
	
	 return usec;
}
#endif







void dofft_r2c_inplace(double *d_data , int DATA_SIZE,int batch,int nLayer){
    int n[1]={DATA_SIZE};
    int inembeb[1]={DATA_SIZE+2};
    int onembeb[1]={(DATA_SIZE+2)/2};
    cufftPlanMany(&plan_dft_r2c_nocubic,1,n,
                    inembeb,1,DATA_SIZE+2,
                    onembeb,1,(DATA_SIZE+2)/2,
                    CUFFT_D2Z, (batch+2)*(nLayer+2));

    cufftExecD2Z(plan_dft_r2c_nocubic, reinterpret_cast<double *>(d_data),
                                reinterpret_cast<cufftDoubleComplex *>(d_data));
  



}
void dofft_c2r_inplace(double *d_data , int DATA_SIZE,int batch,int nLayer){
    int n[1]={DATA_SIZE};
    int inembeb[1]={(DATA_SIZE+2)/2};
    int onembeb[1]={(DATA_SIZE+2)};
    cufftPlanMany(&plan_dft_c2r_nocubic,1,n,
                    inembeb,1,(DATA_SIZE+2)/2,
                    onembeb,1,(DATA_SIZE+2),
                    CUFFT_Z2D, (batch+2)*(nLayer+2));
    
    cufftExecZ2D(plan_dft_c2r_nocubic, reinterpret_cast<cufftDoubleComplex *>(d_data),
                                reinterpret_cast<double *>(d_data));
  



}



void run_dft_r2c_inplace(double *d_data , int DATA_SIZE,cufftHandle &plan_dft_r2c_cubic){

    #ifdef TIME_TEST
    time_postOp=0;
    time_preOp=0;
    time_trans_xzy=0;
    time_trans_zyx=0;
    time_cufft=0;
    #endif



    #ifdef TIME_TEST
    struct timeval tBegin2;
    timeBegin(&tBegin2);
    #endif

    
    cufftExecD2Z(plan_dft_r2c_cubic, reinterpret_cast<double *>(d_data),
                                reinterpret_cast<cufftDoubleComplex *>(d_data));
    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    time_cufft += timeEnd(tBegin2);
    #endif 





    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    printf("timepreOp:    count=3  totaltime=%lld  avetime=%lld \n",time_preOp,time_preOp/3);
    printf("timepostOp:   count=3  totaltime=%lld  avetime=%lld \n",time_postOp,time_postOp/3);
    printf("timecufft:    count=3  totaltime=%lld  avetime=%lld \n",time_cufft,time_cufft/3);
    printf("timetransxzy: count=2  totaltime=%lld  avetime=%lld \n",time_trans_xzy,time_trans_xzy/2);
    printf("timetranszyx: count=2  totaltime=%lld  avetime=%lld \n",time_trans_zyx,time_trans_zyx/2);
    #endif


}
void run_dft_c2r_inplace(double *d_data , int DATA_SIZE,cufftHandle &plan_dft_c2r_cubic){

    #ifdef TIME_TEST
    time_postOp=0;
    time_preOp=0;
    time_trans_xzy=0;
    time_trans_zyx=0;
    time_cufft=0;
    #endif



    #ifdef TIME_TEST
    struct timeval tBegin2;
    timeBegin(&tBegin2);
    #endif

    
    cufftExecZ2D(plan_dft_c2r_cubic, reinterpret_cast<cufftDoubleComplex *>(d_data),
                                reinterpret_cast<double *>(d_data));
    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    time_cufft += timeEnd(tBegin2);
    #endif 





    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    printf("timepreOp:    count=3  totaltime=%lld  avetime=%lld \n",time_preOp,time_preOp/3);
    printf("timepostOp:   count=3  totaltime=%lld  avetime=%lld \n",time_postOp,time_postOp/3);
    printf("timecufft:    count=3  totaltime=%lld  avetime=%lld \n",time_cufft,time_cufft/3);
    printf("timetransxzy: count=2  totaltime=%lld  avetime=%lld \n",time_trans_xzy,time_trans_xzy/2);
    printf("timetranszyx: count=2  totaltime=%lld  avetime=%lld \n",time_trans_zyx,time_trans_zyx/2);
    #endif


}

//这里先默认传入的矩阵每行是DATA_SIZE+2的，后续在考虑是否要进行填充
//DATA_SIZE是需要做fft的数组的长度，传入的长度一般是+2的，batch是这一个平面上有多少个向量，nlayer有多少层，一般也都是要+2的
void run_dft_r2c_inplace_nocubic(double *d_data , int DATA_SIZE,int batch,int nLayer){
    

    #ifdef TIME_TEST
    time_postOp=0;
    time_preOp=0;
    time_trans_xzy=0;
    time_trans_zyx=0;
    time_cufft=0;
    #endif
    
    
    #ifdef TIME_TEST
    struct timeval tBegin2;
    timeBegin(&tBegin2);
    #endif

    dofft_r2c_inplace(d_data,DATA_SIZE,batch,nLayer);
    
    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    time_cufft += timeEnd(tBegin2);
    #endif 

    


    

    
    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    printf("timepreOp:    count=3  totaltime=%lld  avetime=%lld \n",time_preOp,time_preOp/3);
    printf("timepostOp:   count=3  totaltime=%lld  avetime=%lld \n",time_postOp,time_postOp/3);
    printf("timecufft:    count=3  totaltime=%lld  avetime=%lld \n",time_cufft,time_cufft/3);
    printf("timetransxzy: count=2  totaltime=%lld  avetime=%lld \n",time_trans_xzy,time_trans_xzy/2);
    printf("timetranszyx: count=2  totaltime=%lld  avetime=%lld \n",time_trans_zyx,time_trans_zyx/2);
    #endif

    // cudaMemcpy(in, d_data, arraySize,cudaMemcpyDeviceToHost);



    
    freeMemory_r2c();
}
void run_dft_c2r_inplace_nocubic(double *d_data , int DATA_SIZE,int batch,int nLayer){
    

    #ifdef TIME_TEST
    time_postOp=0;
    time_preOp=0;
    time_trans_xzy=0;
    time_trans_zyx=0;
    time_cufft=0;
    #endif
    
    
    #ifdef TIME_TEST
    struct timeval tBegin2;
    timeBegin(&tBegin2);
    #endif

    dofft_c2r_inplace(d_data,DATA_SIZE,batch,nLayer);
    
    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    time_cufft += timeEnd(tBegin2);
    #endif 

    


    

    
    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    printf("timepreOp:    count=3  totaltime=%lld  avetime=%lld \n",time_preOp,time_preOp/3);
    printf("timepostOp:   count=3  totaltime=%lld  avetime=%lld \n",time_postOp,time_postOp/3);
    printf("timecufft:    count=3  totaltime=%lld  avetime=%lld \n",time_cufft,time_cufft/3);
    printf("timetransxzy: count=2  totaltime=%lld  avetime=%lld \n",time_trans_xzy,time_trans_xzy/2);
    printf("timetranszyx: count=2  totaltime=%lld  avetime=%lld \n",time_trans_zyx,time_trans_zyx/2);
    #endif

    // cudaMemcpy(in, d_data, arraySize,cudaMemcpyDeviceToHost);



    
    freeMemory_c2r();
}




void freeMemory_r2c(){
    cufftDestroy(plan_dft_r2c_nocubic);

}
void freeMemory_c2r(){
    cufftDestroy(plan_dft_c2r_nocubic);

}