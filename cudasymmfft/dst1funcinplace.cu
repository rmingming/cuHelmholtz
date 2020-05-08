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




cufftHandle plan_dst1_nocubic;


#ifdef TIME_TEST
static long long time_preOp, time_postOp,time_trans_xzy,time_trans_zyx,time_cufft;
long long tmp;


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




//每个block负责一行的计算


__global__ void preOp_dst1_inplace(double* in,int N,int batch){
    int ibx = blockIdx.x;
    int iby = blockIdx.y;
    double *pin=in+ibx*(N+2)+iby*(N+2)*(batch+2);
    int itx=threadIdx.x;
    extern __shared__ double sh_in[];

    if(itx<N/2){
        sh_in[itx]=pin[itx];
        sh_in[itx+N/2]=pin[itx+N/2];
        
        
    }
    __syncthreads();
    if(itx<N/2+1){
        if(itx==0){
            pin[0]=2*sh_in[1];
            pin[1]=0;
        }else if(itx==N/2){
            pin[N]=-2*sh_in[N-1];
            pin[N+1]=0;
        }else{
            pin[itx*2]=(sh_in[itx*2+1]-sh_in[itx*2-1]);
            pin[itx*2+1]=-(sh_in[itx*2]);
        }
    }
    

}

__global__ void postOp_dst1_inplace(double* in,int N,int batch){
    int ibx = blockIdx.x;
    int iby = blockIdx.y;
    double *pin=in+ibx*(N+2)+iby*(N+2)*(batch+2);
    int itx=threadIdx.x;
    // extern __shared__ double sh_in[];
    // if(itx<N/2){
    //     sh_in[itx]=pin[itx];
    //     sh_in[itx+N/2]=pin[itx+N/2];
    // }
    __syncthreads();

    if(itx<N/2+1){
        if(itx!=0){
            double sitx = pin[itx];
            double sNitx = pin[N-itx];
            double Ta=(sitx+sNitx)/(4*sin(itx*M_PI/N));
            double Tb=(sitx-sNitx)/2;
            __syncthreads();
            pin[itx]=(Ta+Tb)/2;
            pin[N-itx]=(Ta-Tb)/2;
        }else{
            pin[0]=0;
        }
    }
    

}





void dofft_dst1(double *d_data , int DATA_SIZE,int batch,int nLayer){
    int n[1]={DATA_SIZE};
    int inembeb[1]={(DATA_SIZE+2)/2};
    int onembeb[1]={(DATA_SIZE+2)};
    cufftResult r = cufftPlanMany(&plan_dst1_nocubic,1,n,
                    inembeb,1,(DATA_SIZE+2)/2,
                    onembeb,1,(DATA_SIZE+2),
                    CUFFT_Z2D, (batch+2)*(nLayer+2));
    
    if(r!=0){
        printf("CUFFT FAILED! ERROR CODE: %s\n",cufftresultcode[r]);
        exit(0);
    }
    
    cufftExecZ2D(plan_dst1_nocubic, reinterpret_cast<cufftDoubleComplex *>(d_data),
                                reinterpret_cast<double *>(d_data));
  
    // printf("datasize=%d,r=%d\n",DATA_SIZE,r);


}


//这里先默认传入的矩阵每行是DATA_SIZE+2的，后续在考虑是否要进行填充
//DATA_SIZE是需要做fft的数组的长度，传入的长度一般是+2的，batch是这一个平面上有多少个向量，nlayer有多少层，一般也都是要+2的
void run_3d_dst_1_inplace_nocubic(double *d_data , int DATA_SIZE,int batch,int nLayer){
    // printf("%d %d %d\n",DATA_SIZE,batch,nLayer);
    #ifdef TIME_TEST
    time_postOp=0;
    time_preOp=0;
    time_trans_xzy=0;
    time_trans_zyx=0;
    time_cufft=0;
    #endif
    
    dim3 preOpGridDim;
    preOpGridDim.x=batch+1;
    preOpGridDim.y=nLayer+1;
    preOpGridDim.z=1;

    #ifdef TIME_TEST
    struct timeval tBegin1;
    timeBegin(&tBegin1);
    
    
    #endif
    preOp_dst1_inplace<<<preOpGridDim,DATA_SIZE/2+1,sizeof(double)*DATA_SIZE>>>(d_data,DATA_SIZE,batch);
    // preOp_dst1_inplace<<<preOpGridDim,DATA_SIZE/2,sizeof(double)*DATA_SIZE>>>(d_data,DATA_SIZE,batch);
    
    #ifdef TIME_TEST
    
    cudaDeviceSynchronize();
    time_preOp = timeEnd(tBegin1);
    
    #endif 
    
    #ifdef TIME_TEST
    struct timeval tBegin2;
    timeBegin(&tBegin2);
    #endif

    dofft_dst1(d_data,DATA_SIZE,batch,nLayer);
  

    

    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    time_cufft += timeEnd(tBegin2);
    #endif 

    


    dim3 postOpGridDim;
    postOpGridDim.x=batch+1;
    postOpGridDim.y=nLayer+1;
    postOpGridDim.z=1;

    #ifdef TIME_TEST
    struct timeval tBegin3;
    timeBegin(&tBegin3);
    #endif
    postOp_dst1_inplace<<<postOpGridDim,DATA_SIZE/2+1>>>(d_data,DATA_SIZE,batch);
    // postOp_dst1_inplace<<<postOpGridDim,DATA_SIZE/2,sizeof(double)*DATA_SIZE>>>(d_data,DATA_SIZE,batch);

    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    time_postOp += timeEnd(tBegin3);
    #endif 


    
    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    printf("timepreOp:    count=3  totaltime=%lld  avetime=%lld \n",time_preOp,time_preOp/3);
    printf("timepostOp:   count=3  totaltime=%lld  avetime=%lld \n",time_postOp,time_postOp/3);
    printf("timecufft:    count=3  totaltime=%lld  avetime=%lld \n",time_cufft,time_cufft/3);
    printf("timetransxzy: count=2  totaltime=%lld  avetime=%lld \n",time_trans_xzy,time_trans_xzy/2);
    printf("timetranszyx: count=2  totaltime=%lld  avetime=%lld \n",time_trans_zyx,time_trans_zyx/2);
    #endif
    



    
    freeMemory_dst1();
}


void run_3d_dst_1_inplace(double *d_data , int DATA_SIZE,cufftHandle &plan_dst1_cubic){
    #ifdef TIME_TEST
    time_postOp=0;
    time_preOp=0;
    time_trans_xzy=0;
    time_trans_zyx=0;
    time_cufft=0;
    #endif

    dim3 preOpGridDim;
    preOpGridDim.x=DATA_SIZE+1;
    preOpGridDim.y=DATA_SIZE+1;
    preOpGridDim.z=1;

    #ifdef TIME_TEST
    struct timeval tBegin1;
    timeBegin(&tBegin1);

    
    #endif
    preOp_dst1_inplace<<<preOpGridDim,DATA_SIZE/2+1,sizeof(double)*DATA_SIZE>>>(d_data,DATA_SIZE,DATA_SIZE);
    // preOp_dst1_inplace<<<preOpGridDim,DATA_SIZE/2,sizeof(double)*DATA_SIZE>>>(d_data,DATA_SIZE,DATA_SIZE);
    
    #ifdef TIME_TEST

    cudaDeviceSynchronize();
    time_preOp = timeEnd(tBegin1);

    #endif 

    #ifdef TIME_TEST
    struct timeval tBegin2;
    timeBegin(&tBegin2);
    #endif

    // cufftExecD2Z(plan_dst1_cubic, reinterpret_cast<double *>(d_data),
    //                             reinterpret_cast<cufftDoubleComplex *>(d_data));
    cufftExecZ2D(plan_dst1_cubic, reinterpret_cast<cufftDoubleComplex*>(d_data),
                                reinterpret_cast<double *>(d_data));
  

   

    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    time_cufft += timeEnd(tBegin2);
    #endif 

    


    dim3 postOpGridDim;
    postOpGridDim.x=DATA_SIZE+1;
    postOpGridDim.y=DATA_SIZE+1;
    postOpGridDim.z=1;

    #ifdef TIME_TEST
    struct timeval tBegin3;
    timeBegin(&tBegin3);
    #endif
    postOp_dst1_inplace<<<postOpGridDim,DATA_SIZE/2+1>>>(d_data,DATA_SIZE,DATA_SIZE);
    // postOp_dst1_inplace<<<postOpGridDim,DATA_SIZE/2,sizeof(double)*DATA_SIZE>>>(d_data,DATA_SIZE,DATA_SIZE);

    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    time_postOp += timeEnd(tBegin3);
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


void freeMemory_dst1(){

    cufftDestroy(plan_dst1_nocubic);


}
