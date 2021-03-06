#include "hw3crt_wrapper.h"
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "time_.h"
#include "ufunc.h"

#define PI 3.14159265358979323846
//#define DEBUG

using namespace std;

//in this code, we assume that solution domain is [0,1]^3
int main(int argc, char **argv)
{
	if (argc < 5)
	{
		printf("usage: ./testcpu N xbc ybc zbc\n");
		exit(-1);
	}
	int N = atoi(argv[1]);
	int xbc = atoi(argv[2]);
	int ybc = atoi(argv[3]);
	int zbc = atoi(argv[4]);
	int nx=N; 
	int ny=N;
	int nz=N;
	double lambda = -1; 
	int i, j, k;
	double dx = 1.0/nx;
	double dy = 1.0/ny;
	double dz = 1.0/nz;
	int sz = (nx+1)*(ny+1)*(nz+1);
	double *f =  new double[sz];
	double *bcl=NULL;
	double *bcr=NULL;
	double *bcb=NULL;
	double *bct=NULL;
	double *bcf=NULL;
	double *bce=NULL;

	if ( xbc == 3 || xbc == 4 )
		bcl = (double *) malloc(sizeof(double)*(ny+1)*(nz+1));
	if ( xbc == 3 || xbc == 2 )
		bcr = (double *) malloc(sizeof(double)*(ny+1)*(nz+1));
	if ( ybc == 3 || ybc == 4 )
		bcb = (double *) malloc(sizeof(double)*(nx+1)*(nz+1));
	if ( ybc == 3 || ybc == 2 )
		bct = (double *) malloc(sizeof(double)*(nx+1)*(nz+1));
	if ( zbc == 3 || zbc == 4 )
		bcf = (double *) malloc(sizeof(double)*(nx+1)*(ny+1));
	if ( zbc == 3 || zbc == 2 )
		bce = (double *) malloc(sizeof(double)*(nx+1)*(ny+1));

	//assign right hand side
	for (k=0; k<nz+1; k++)
		for (j=0; j<ny+1; j++)
			for (i=0; i<nx+1; i++)
				f[k*(nx+1)*(ny+1)+j*(nx+1)+i] = 
					laplaceufunc(i*dx,j*dy,k*dz)+lambda*ufunc(i*dx,j*dy,k*dz);

	//assign x boundary condition
	//for Dirichlet boundary condition
	for (k=0; k<nz+1; k++)
		for (j=0; j<ny+1; j++)
		{
			if ( xbc == 1 || xbc == 2 )
				f[k*(nx+1)*(ny+1)+j*(nx+1)+0] = ufunc(0*dx,j*dy,k*dz);
			if ( xbc == 1 || xbc == 4 )
				f[k*(nx+1)*(ny+1)+j*(nx+1)+nx] = ufunc(nx*dx,j*dy,k*dz);
		}
	//for Neumann boundary condition
	for (k=0; k<nz+1; k++)
		for (j=0; j<ny+1; j++)
		{
			if ( xbc == 3 || xbc == 4 )
				bcl[k*(ny+1)+j] = uxfunc(0*dx,j*dy,k*dz);
			if ( xbc == 3 || xbc == 2 )
				bcr[k*(ny+1)+j] = uxfunc(nx*dx,j*dy,k*dz);
		}

	//assign y boundary condition
	//for Dirichlet boundary condition
	for (k=0; k<nz+1; k++)
		for (i=0; i<nx+1; i++)
		{
			if ( ybc == 1 || ybc == 2 )
				f[k*(nx+1)*(ny+1)+0*(nx+1)+i] = ufunc(i*dx,0*dy,k*dz);
			if ( ybc == 1 || ybc == 4 )
				f[k*(nx+1)*(ny+1)+ny*(nx+1)+i] = ufunc(i*dx,ny*dy,k*dz);
		}
	//for Neumann boundary condition
	for (k=0; k<nz+1; k++)
		for (i=0; i<nx+1; i++)
		{
			if ( ybc == 3 || ybc == 4 )
				bcb[k*(nx+1)+i] = uyfunc(i*dx,0*dy,k*dz);
			if ( ybc == 3 || ybc == 2 )
				bct[k*(nx+1)+i] = uyfunc(i*dx,ny*dy,k*dz);
		}

	//assign z boundary condition
	//for Dirichlet boundary condition
	for (j=0; j<ny+1; j++)
		for (i=0; i<nx+1; i++)
		{
			if ( zbc == 1 || zbc == 2 )
				f[0*(nx+1)*(ny+1)+j*(nx+1)+i] = ufunc(i*dx,j*dy,0*dz);
			if ( zbc == 1 || zbc == 4 )
				f[nz*(nx+1)*(ny+1)+j*(nx+1)+i] = ufunc(i*dx,j*dy,nz*dz);
		}
	//for Neumann boundary condition
	for (j=0; j<ny+1; j++)
		for (i=0; i<nx+1; i++)
		{
			if ( zbc == 3 || zbc == 4 )
				bcf[j*(nx+1)+i] = uzfunc(i*dx,j*dy,0*dz);
			if ( zbc == 3 || zbc == 2 )
				bce[j*(nx+1)+i] = uzfunc(i*dx,j*dy,nz*dz);
		}

	time_(
	solver_3d(0, 1, nx, xbc, bcl, bcr, 
			  0, 1, ny, ybc, bcb, bct,  
			  0, 1, nz, zbc, bcf, bce, 
			  -1, f);
	)

//	printf("***********************************\n");

	double maxim = 0;
	for (k=0; k<=nz; k++)
	{
		for (j=0; j<=ny; j++)
		{
			for (i=0; i<=nx; i++)
			{
				double dif = f[k*(nx+1)*(ny+1)+j*(nx+1)+i] - ufunc(i*dx,j*dy,k*dz);
#ifdef DEBUG
				printf("%12.8f", dif);
#endif
				dif = fabs(dif);
				if (maxim < dif)
					maxim = dif;
#ifdef DEBUG
				if (i%9 == 8)
					printf("\n");
#endif
			}
		}
#ifdef DEBUG
		printf("\n");
#endif
	}

	printf("max diff : %.14f\n", maxim);

	delete [] f;
	if ( xbc == 3 || xbc == 4 )
		free(bcl);
	if ( xbc == 3 || xbc == 2 )
		free(bcr);
	if ( ybc == 3 || ybc == 4 )
		free(bcb);
	if ( ybc == 3 || ybc == 2 )
		free(bct);
	if ( zbc == 3 || zbc == 4 )
		free(bcf);
	if ( zbc == 3 || zbc == 2 )
		free(bce);

	return 0;
}
