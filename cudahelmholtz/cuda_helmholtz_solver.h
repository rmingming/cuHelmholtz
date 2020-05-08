#pragma once
/*
   laplace u + lambda * u = f
*/
int cuda_helmholtz_solver(double left, double right, int NX, int xbc, 
		double *bcl, double *bcr, 
		double bottom, double top, int NY, int ybc, 
		double *bcb, double *bct, 
		double front, double back, int NZ, int zbc, 
		double *bcf, double *bce, // bce = bc back
		double lambda, double * f);
