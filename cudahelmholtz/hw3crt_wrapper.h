extern "C" {
	void hw3crt_( 
			double * xs, double * xf, int * l, int * lbdcnd, double * bdxs, double * bdxf,
			double * ys, double * yf, int * m, int * mbdcnd, double * bdys, double * bdyf,
			double * zs, double * zf, int * n, int * nbdcnd, double * bdzs, double * bdzf,
			double * elmbda, int * ldimf, int * mdimf, double * f, 
			double * pertrb, int * ierror, double * w
			);
}



void solver_3d(double xs, double xf, int NX, int xbc, 
		double *bcxs, double *bcxf, 
		double ys, double yf, int NY, int ybc, 
		double *bcys, double *bcyf, 
		double zs, double zf, int NZ, int zbc, 
		double *bczs, double *bczf, 
		double lambda, double * f);
