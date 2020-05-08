# cuHelmholtz
## intro
a CUDA based Library for Discrete Sine/Cosine Transform and Solution of Helmholtz Equation

## install 
this project use CUDA and GCC. Go check them out if you don't have them locally installed.

```bash
cd likefftwall 
make 
make lib
```

## usage
After installation, you can find the library files libcudasymmfft.a in this folder

Users must include the header ```funcinterface.h``` and link the lib ```libcudasymmfft.a```

Users can call the following API to complete the DST and DCT calculation
```cpp
void do_transform(double *d_data,               //input data
                    double *d_out,              //output(In-place calculation when equal to input data)
                    int NX,                     //size of X dim
                    int NY,                     //size of Y dim
                    int NZ,                     //size of Z dim
                    transform_kind X_TRANS,     //transform type of X
                    transform_kind Y_TRANS,     //transform type of Y
                    transform_kind Z_TRANS);    //transform type of Z
```

Note that the NX, NY and NZ must be equal for in-place calculations.