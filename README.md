# cuHelmholtz
## Introduction
* ```cudasymmfft```
A CUDA based Library for three-dimensional Discrete Fourier, Sine and Cosine Transform (DFT, DST, DCT).
* ```cudahelmholtz```
A GPU solver for Helmholtz Equation by using cudasymmfft library.


## cudasymmfft
### Installation
To compiler this library you need to use CUDA and GCC. Go check them out if you don't have them locally installed. We have tested on CUDA 6.0 and GCC 4.8.4.

```bash
cd cudasymmfft 
make 
make lib
```

### Usage
After installation, you can find the library files ```libcudasymmfft.a``` in this folder

You must include the header ```funcinterface.h``` and link against the lib ```libcudasymmfft.a```

You can call the following API to complete the three-dimensional mixed DFT, DST and DCT calculation.
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

## cudahelmholtz
### Introduction
```cudahelmholtz``` solves Helmholtz equation ```laplace u + lambda * u = f``` on GPU. It can solve three-dimensional Helmholtz equations with different boundary conditions. The boundary condition in each direction can be periodic, Dirichlet or Neumann, and ```cudahelmholtz``` allows different boundary conditions in the three (x, y and z) directions.


### Requirement
We use ```cuTranspose``` in ```cudahelmholtz```. You should install it first.

### Usage
```bash
cd cudahelmholtz 
make testgpu
./testgpu 128 1 2 0 
```
In the above line ```./testgpu 128 1 2 0```, 128 means the number of cells in each direction. If you want to use different cells in different directions, you should use ```make testgpurectangle``` instead of ```make testgpu```. 
The three numbers after 128 are boundary condtions in x, y and z directions respectively. Each number can be 0, 1, 2, 3, and 4 which has the following meaning:
* 0 means periodic
* 1 means left and right are both Dirichlet boundaries
* 2 means left is Dirichlet boundary and right is Neumann boundary
* 3 means left and right are both Neumann boundaries
* 4 means left is Neumann boundary and right is Dirichlet boundary. 

### Caution
If you want to change the testing function, you just need to modify file ```ufunc.h```. But pay attention to the periodic boundary condition. When you want to use periodic boundary condition, you must make sure that the inline functions in file ```ufunc.h``` return identical values at the two ends in the periodic direction. We have provided three testing examples in this file.
