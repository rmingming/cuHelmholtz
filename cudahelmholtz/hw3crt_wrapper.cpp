#include "hw3crt_wrapper.h"
#include <iostream>
#include <algorithm>

void solver_3d(double xs, double xf, int NX, int xbc, 
		double *bcxs, double *bcxf, 
		double ys, double yf, int NY, int ybc, 
		double *bcys, double *bcyf, 
		double zs, double zf, int NZ, int zbc, 
		double *bczs, double *bczf, 
		double lambda, double * f)
{
	int l = NX ;
	int m = NY ;
	int n = NZ ;
	int ldimf = l + 1 ;
	int mdimf = m + 1 ;
	int ierror = -1; // initial value, output would be ge. 0;
	double pertrb;
	int wsize = 30 + l + m + 5 * n + std::max(std::max(l,m),n) + 7 * ((l+1)/2 + (m+1)/2);
	double * w = new double[wsize];

	hw3crt_( 
			&xs, &xf, &l, &xbc, bcxs, bcxf, 
			&ys, &yf, &m, &ybc, bcys, bcyf, 
			&zs, &zf, &n, &zbc, bczs, bczf, 
			&lambda, &ldimf, &mdimf, f,
			&pertrb, &ierror, w
		   );

	std::cout << "State code : " << ierror << std::endl;
	delete [] w;
}

