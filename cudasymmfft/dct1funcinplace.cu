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


//#define TIME_TEST


double *d_x1;

cufftHandle plan_dct1_nocubic;


#ifdef TIME_TEST
static long long time_preOp, time_postOp,time_trans_xzy,time_trans_zyx,time_cufft;

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

int pow2roundup (int x){
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}


__global__ void preOp_dct1_inplace(double* in,int N,double *x1,int nThread,int batch){
    int ibx = blockIdx.x;
    int iby = blockIdx.y;
    double *pin=in+ibx*(N+2)+iby*(N+2)*(batch+1);
    double *px1=x1+ibx+iby*(batch+1);
    int itx=threadIdx.x;
    extern __shared__ double sh_in[];

    if(itx<N/2){
        sh_in[itx]=pin[itx*2+1];
    }else{
        sh_in[itx]=0;
    }
    __syncthreads();
    if(itx<N/2){
        if(itx==0){
            pin[N+1]=0;
            pin[1]=0;
        }else{
            pin[itx*2+1]=((sh_in[itx-1]-sh_in[itx]));
        }
    }
    __syncthreads();
    for(unsigned int s=nThread>>1;s>0;s>>=1){
        if(itx<s)
            sh_in[itx]+=sh_in[itx+s];
        // if((itx)*2+1+i<N&&(itx)%(i)==0){
        //     sh_in[(itx)*2+1]+=sh_in[(itx)*2+1+i];
        // }
        __syncthreads();
    }
    if(itx==0){
        px1[0]=sh_in[0];
        // printf("\nx1=%f\n",px1[0]);
    }
    // __syncthreads();

}



__global__ void postOp_dct1_inplace(double* in,int N,double *x1,int batch){
    int itx=threadIdx.x;
    int ibx = blockIdx.x;
    int iby = blockIdx.y;
    double *pin=in+ibx*(N+2)+iby*(N+2)*(batch+1);
    double *px1=x1+ibx+iby*(batch+1);
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
            double Ta=(sitx+sNitx)*0.5;
            double Tb=(sitx-sNitx)*0.25/sin(itx*M_PI/N);
            // __syncthreads();
            pin[itx]=(Ta-Tb)*0.5;
            pin[N-itx]=(Ta+Tb)*0.5;
        }else{
            double sh0=pin[0]*0.5;
            pin[0]=sh0+px1[0];
            pin[N]=sh0-px1[0];
        }
    }
    

}

void dofft_dct1(double *d_data , int DATA_SIZE,int batch,int nLayer){
    int n[1]={DATA_SIZE-1};
    int inembeb[1]={(DATA_SIZE+1)/2};
    int onembeb[1]={(DATA_SIZE+1)};
    cufftResult r = cufftPlanMany(&plan_dct1_nocubic,1,n,
                    inembeb,1,(DATA_SIZE+1)/2,
                    onembeb,1,(DATA_SIZE+1),
                    CUFFT_Z2D, (batch+1)*(nLayer+1));
    
    if(r!=0){
        printf("CUFFT FAILED! ERROR CODE: %s\n",cufftresultcode[r]);
        exit(0);
    }                
    cufftExecZ2D(plan_dct1_nocubic, reinterpret_cast<cufftDoubleComplex *>(d_data),
                                reinterpret_cast<double *>(d_data));
  
}
//这里的DATA_SIZE是2^n+1
// void dofft_dct1(double *d_data , int DATA_SIZE,int batch,int nLayer){
//     int n[1]={DATA_SIZE-1};
//     int inembeb[1]={DATA_SIZE+1};
//     int onembeb[1]={(DATA_SIZE+1)/2};
//     cufftPlanMany(&plan_dct1_nocubic,1,n,
//                     inembeb,1,DATA_SIZE+1,
//                     onembeb,1,(DATA_SIZE+1)/2,
//                     CUFFT_D2Z, (batch+1)*(nLayer+1));
    
//     cufftExecD2Z(plan_dct1_nocubic, reinterpret_cast<double *>(d_data),
//                                 reinterpret_cast<cufftDoubleComplex *>(d_data));
  

//     // if (cudaDeviceSynchronize() != cudaSuccess){
//     //     printf("Cuda error: Failed to synchronize\n"); 	
//     //     return;
//     // }

    
// }


//这里先默认传入的矩阵每行是DATA_SIZE+2的，后续在考虑是否要进行填充
//DATA_SIZE是需要做fft的数组的长度，传入的长度一般是+2的
//这里的DATA_SIZE是2^N+1
void run_3d_dct_1_inplace_nocubic(double *d_data , int DATA_SIZE,int batch,int nLayer){
    


    #ifdef TIME_TEST
    time_postOp=0;
    time_preOp=0;
    time_trans_xzy=0;
    time_trans_zyx=0;
    time_cufft=0;
    #endif


    cudaMalloc(reinterpret_cast<void **>(&d_x1), sizeof(double)*(batch+1)*(nLayer+1));


    int nThread = pow2roundup((DATA_SIZE-1)/2);
    
    dim3 preOpGridDim;
    preOpGridDim.x=batch;
    preOpGridDim.y=nLayer;
    preOpGridDim.z=1;

    #ifdef TIME_TEST
    struct timeval tBegin1;
    timeBegin(&tBegin1);
    #endif
    preOp_dct1_inplace<<<preOpGridDim,nThread,sizeof(double)*nThread>>>(d_data,DATA_SIZE-1,d_x1,nThread,batch);
    // preOp_dct1_inplace<<<preOpGridDim,DATA_SIZE+1,sizeof(double)*DATA_SIZE>>>(d_data,DATA_SIZE-1,d_x1,batch);

    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    time_preOp = timeEnd(tBegin1);
    #endif 
    


    #ifdef TIME_TEST
    struct timeval tBegin2;
    timeBegin(&tBegin2);
    #endif

    dofft_dct1(d_data,DATA_SIZE,batch,nLayer);

    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    time_cufft += timeEnd(tBegin2);
    #endif 
 


    dim3 postOpGridDim;
    postOpGridDim.x=batch;
    postOpGridDim.y=nLayer;
    postOpGridDim.z=1;

    #ifdef TIME_TEST
    struct timeval tBegin3;
    timeBegin(&tBegin3);
    #endif
    postOp_dct1_inplace<<<postOpGridDim,DATA_SIZE/2+1>>>(d_data,DATA_SIZE-1,d_x1,batch);
    // postOp_dct1_inplace<<<postOpGridDim,DATA_SIZE/2+1,sizeof(double)*DATA_SIZE/2>>>(d_data,d_x1,DATA_SIZE-1,batch);

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
    

    
    freeMemory_dct1_nocubic();
}

void run_3d_dct_1_inplace(double *d_data , int DATA_SIZE,cufftHandle &plan_dct1_cubic){
    

    

    #ifdef TIME_TEST
    time_postOp=0;
    time_preOp=0;
    time_trans_xzy=0;
    time_trans_zyx=0;
    time_cufft=0;
    #endif

    
    cudaMalloc(reinterpret_cast<void **>(&d_x1), sizeof(double)*(DATA_SIZE+1)*(DATA_SIZE+1));
    


    int nThread = pow2roundup((DATA_SIZE-1)/2);

    dim3 preOpGridDim;
    preOpGridDim.x=DATA_SIZE;
    preOpGridDim.y=DATA_SIZE;
    preOpGridDim.z=1;

    #ifdef TIME_TEST
    struct timeval tBegin1;
    timeBegin(&tBegin1);
    #endif
    preOp_dct1_inplace<<<preOpGridDim,nThread,sizeof(double)*nThread>>>(d_data,DATA_SIZE-1,d_x1,nThread,DATA_SIZE);
    // preOp_dct1_inplace<<<preOpGridDim,DATA_SIZE+1,sizeof(double)*DATA_SIZE>>>(d_data,DATA_SIZE-1,d_x1,DATA_SIZE);

    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    time_preOp = timeEnd(tBegin1);
    #endif 
    


    #ifdef TIME_TEST
    struct timeval tBegin2;
    timeBegin(&tBegin2);
    #endif

    
    // cufftExecD2Z(plan_dct1_cubic, reinterpret_cast<double *>(d_data),
    //                 reinterpret_cast<cufftDoubleComplex *>(d_data));
    
    cufftExecZ2D(plan_dct1_cubic, reinterpret_cast<cufftDoubleComplex *>(d_data),
                    reinterpret_cast<double *>(d_data));
    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    time_cufft += timeEnd(tBegin2);
    #endif 
 


    dim3 postOpGridDim;
    postOpGridDim.x=DATA_SIZE;
    postOpGridDim.y=DATA_SIZE;
    postOpGridDim.z=1;

    #ifdef TIME_TEST
    struct timeval tBegin3;
    timeBegin(&tBegin3);
    #endif

    postOp_dct1_inplace<<<postOpGridDim,DATA_SIZE/2+1>>>(d_data,DATA_SIZE-1,d_x1,DATA_SIZE);
    // postOp_dct1_inplace<<<postOpGridDim,DATA_SIZE/2+1,sizeof(double)*DATA_SIZE/2>>>(d_data,d_x1,DATA_SIZE-1,DATA_SIZE);

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
    

    
    freeMemory_dct1_cubic();
}



void freeMemory_dct1_cubic(){

    cudaFree(d_x1);

}


void freeMemory_dct1_nocubic(){
    
    cufftDestroy(plan_dct1_nocubic);
    cudaFree(d_x1);

}
