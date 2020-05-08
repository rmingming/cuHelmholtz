#ifndef __GLOBAL__

#define __GLOBAL__

#define TRANS_BLOCK_DIM 16
enum transform_kind {
    DCT_1,
    DCT_2,
    DCT_3,
    DST_1,
    DST_2,
    DST_3,
    DFT_R2C,
    DFT_C2R
};

static char* cufftresultcode[17]={
    "CUFFT_SUCCESS"        ,  //  The cuFFT operation was successful
    "CUFFT_INVALID_PLAN "  ,  //  cuFFT was passed an invalid plan handle
    "CUFFT_ALLOC_FAILED"   ,  //  cuFFT failed to allocate GPU or CPU memory
    "CUFFT_INVALID_TYPE"   ,  //  No longer used
    "CUFFT_INVALID_VALUE"  ,  //  User specified an invalid pointer or parameter
    "CUFFT_INTERNAL_ERROR" ,  //  Driver or internal cuFFT library error
    "CUFFT_EXEC_FAILED"    ,  //  Failed to execute an FFT on the GPU
    "CUFFT_SETUP_FAILED"   ,  //  The cuFFT library failed to initialize
    "CUFFT_INVALID_SIZE"   ,  //  User specified an invalid transform size
    "CUFFT_UNALIGNED_DATA" ,  //  No longer used
    "CUFFT_INCOMPLETE_PARAMETER_LIST" , //  Missing parameters in call
    "CUFFT_INVALID_DEVICE" , //  Execution of a plan was on different GPU than plan creation
    "CUFFT_PARSE_ERROR"    , //  Internal plan database error 
    "CUFFT_NO_WORKSPACE"     //  No workspace has been provided prior to plan execution
    "CUFFT_NOT_IMPLEMENTED" , // Function does not implement functionality for parameters given.
    "CUFFT_LICENSE_ERROR"  , // Used in previous versions.
    "CUFFT_NOT_SUPPORTED"    // Operation is not supported for parameters given.
};

#endif /* __GLOBAL__ */