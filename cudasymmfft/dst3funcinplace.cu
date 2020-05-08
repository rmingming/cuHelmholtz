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




cufftHandle plan_dst3_nocubic;

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

//outN是2的整数次幂，就是处理之后的要进行fft的数组的长度
__global__ void preOp_dst3_inplace(double* in,int outN,int batch){
    int itx = threadIdx.x;
    int ibx = blockIdx.x;
    int iby = blockIdx.y;
    double *pin=in+iby*(outN+2)*(batch+1)+ibx*(outN+2);
    if(itx<outN/2+1){
        double sina;
        double cosa;
        sincos((itx)*M_PI/(2*outN),&sina,&cosa);
        // double sina= sin((itx)*M_PI/(2*outN));
        // double cosa= cos((itx)*M_PI/(2*outN));
        double Ta= (pin[itx]+pin[outN-itx]);
        double Tb= (pin[itx]-pin[outN-itx]);
        // __syncthreads();
        pin[itx] = Ta*cosa+Tb*sina;
        pin[outN-itx]= Ta*sina-Tb*cosa;
        
    }
}
__global__ void postOp_dst3_inplace(double* in,int N,int batch){
    extern __shared__ double sh_in[];
    
    int itx = threadIdx.x;
    int ibx = blockIdx.x;
    int iby = blockIdx.y;
    double *pin=in+iby*(N+2)*(batch+1)+ibx*(N+2);

    if(itx<N/2+1){
        sh_in[itx]=pin[itx];
        sh_in[itx+N/2+1]=pin[itx+N/2+1];
        __syncthreads();
        //因为乘了2还得除2，就都不进行乘2操作了
        //这里bk值得符号也省略掉，在下方运算bk+-ak的时候要注意正负号
        if(itx!=0){
            
            sh_in[itx*2+1]=(-1)*sh_in[itx*2+1];
        }
        __syncthreads();
        
        //与其他的DCT和DST一致，也除以2，这样算出的结果是fftw的1/2

        if(itx==0){
            pin[0]=0;
            pin[1]=sh_in[0]/2;
            
        }else{
            pin[2*itx]=(sh_in[itx*2+1]-sh_in[itx*2])/2;
            
            if(itx*2+1<N+1){
                pin[2*itx+1]=(sh_in[itx*2]+sh_in[itx*2+1])/2;
                
            }
            
        }
    }
    
}


//这里的DATA_SIZE是2^n+1
void dofft_inplace(double *d_data , int DATA_SIZE,int batch,int nLayer){
    int n[1]={DATA_SIZE-1};
    int inembeb[1]={DATA_SIZE+1};
    int onembeb[1]={(DATA_SIZE+1)/2};
    cufftResult r = cufftPlanMany(&plan_dst3_nocubic,1,n,
                    inembeb,1,DATA_SIZE+1,
                    onembeb,1,(DATA_SIZE+1)/2,
                    CUFFT_D2Z, (batch+1)*(nLayer+1));
    
    if(r!=0){
        printf("CUFFT FAILED! ERROR CODE: %s\n",cufftresultcode[r]);
        exit(0);
    }


    cufftExecD2Z(plan_dst3_nocubic, reinterpret_cast<double *>(d_data),
                                reinterpret_cast<cufftDoubleComplex *>(d_data));
  

    

}


//这里先默认传入的矩阵每行是DATA_SIZE+2的，后续在考虑是否要进行填充
//DATA_SIZE是需要做fft的数组的长度，传入的长度一般是+2的
//这里的DATA_SIZE是2^N+1
void run_3d_dst_3_inplace_nocubic(double *d_data , int DATA_SIZE,int batch,int nLayer){

    #ifdef TIME_TEST
    time_postOp=0;
    time_preOp=0;
    time_trans_xzy=0;
    time_trans_zyx=0;
    time_cufft=0;
    #endif
    

    dim3 preOpGridDim;
    preOpGridDim.x=batch;
    preOpGridDim.y=nLayer;
    preOpGridDim.z=1;

    #ifdef TIME_TEST
    struct timeval tBegin1;
    timeBegin(&tBegin1);
    #endif

    preOp_dst3_inplace<<<preOpGridDim,DATA_SIZE/2+1>>>(d_data,DATA_SIZE-1,batch);

    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    time_preOp = timeEnd(tBegin1);
    #endif 
    
    

    #ifdef TIME_TEST
    struct timeval tBegin2;
    timeBegin(&tBegin2);
    #endif

    dofft_inplace(d_data,DATA_SIZE,batch,nLayer);
    

    
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

    postOp_dst3_inplace<<<postOpGridDim,DATA_SIZE/2+1,sizeof(double)*(DATA_SIZE+1)>>>(d_data,DATA_SIZE-1,batch);

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
    

 
    freeMemory_dst3();
}


void run_3d_dst_3_inplace(double *d_data , int DATA_SIZE,cufftHandle &plan_dst3_cubic){

    #ifdef TIME_TEST
    time_postOp=0;
    time_preOp=0;
    time_trans_xzy=0;
    time_trans_zyx=0;
    time_cufft=0;
    #endif


    dim3 preOpGridDim;
    preOpGridDim.x=DATA_SIZE;
    preOpGridDim.y=DATA_SIZE;
    preOpGridDim.z=1;

    #ifdef TIME_TEST
    struct timeval tBegin1;
    timeBegin(&tBegin1);
    #endif

    preOp_dst3_inplace<<<preOpGridDim,DATA_SIZE/2+1>>>(d_data,DATA_SIZE-1,DATA_SIZE);

    #ifdef TIME_TEST
    cudaDeviceSynchronize();
    time_preOp = timeEnd(tBegin1);
    #endif 
    


    #ifdef TIME_TEST
    struct timeval tBegin2;
    timeBegin(&tBegin2);
    #endif

    cufftExecD2Z(plan_dst3_cubic, reinterpret_cast<double *>(d_data),
                reinterpret_cast<cufftDoubleComplex *>(d_data));

    
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

    postOp_dst3_inplace<<<postOpGridDim,DATA_SIZE/2+1,sizeof(double)*(DATA_SIZE+1)>>>(d_data,DATA_SIZE-1,DATA_SIZE);

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


//for nocubic
void freeMemory_dst3(){

    cufftDestroy(plan_dst3_nocubic);


}
