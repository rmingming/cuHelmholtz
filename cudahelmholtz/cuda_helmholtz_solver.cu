#include <cufft.h>
#include "cuda_helmholtz_solver.h"
#include <iostream>
#include "time_.h"
#include "funcinterface.h"
#include "global.h"

#define PI 3.14159265358979323846

#include "_cuda_helmholtz_.inl"

#define ceil_div(x, y) (((x)+(y)-1)/(y))
inline dim3 makegrid(int dbx, int dby, int nx, int ny)
{
	return dim3(ceil_div(nx, dbx), ceil_div(ny, dby));
}
inline dim3 makegrid(int dbx, int dby, int dbz, int nx, int ny, int nz)
{
	return dim3(ceil_div(nx, dbx), ceil_div(ny, dby), ceil_div(nz, dbz));
}

void check_error() {
	cudaThreadSynchronize();
	cudaError_t e = cudaGetLastError();
	if (e==cudaSuccess)
		return;
	std::cout << cudaGetErrorString(e) << std::endl;
	exit(-1);
}
/*
   laplace u + lambda * u = f
*/
int cuda_helmholtz_solver(double left, double right, int NX, int xbc, 
		double *bcl, double *bcr, 
		double bottom, double top, int NY, int ybc, 
		double *bcb, double *bct, 
		double front, double back, int NZ, int zbc, 
		double *bcf, double *bce, // bce = bc back
		double lambda, double * f)
{
	int xs, xe;
	int ys, ye;
	int zs, ze;
	double dx = NX/(right - left); //1/dx actually
	double dy = NY/(top - bottom);
	double dz = NZ/(back - front);
	double dx2 = dx * dx;          // 1/(dx*dx) actually
	double dy2 = dy * dy;
	double dz2 = dz * dz;
	//double perturb;

	int x_alloc_size = 2 * (NX/2 + 1);
	int y_alloc_size = 2 * (NY/2 + 1);
	int z_alloc_size = 2 * (NZ/2 + 1);

	// Transformation size

	double shiftx = 0;
	double shifty = 0;
	double shiftz = 0;
	if (xbc == 2) shiftx = - 0.5 * PI / NX;
	if (xbc == 4) shiftx =   0.5 * PI / NX;
	if (ybc == 2) shifty = - 0.5 * PI / NY;
	if (ybc == 4) shifty =   0.5 * PI / NY;
	if (zbc == 2) shiftz = - 0.5 * PI / NZ;
	if (zbc == 4) shiftz =   0.5 * PI / NZ;

	double pidivnx = PI/NX;
	double pidivny = PI/NY;
	double pidivnz = PI/NZ;
	if (xbc == 0) pidivnx *= 2;
	if (ybc == 0) pidivny *= 2;
	if (zbc == 0) pidivnz *= 2;

	enum transform_kind xkind, ykind, zkind;
	enum transform_kind xkind_inv, ykind_inv, zkind_inv;

	switch (xbc) {
		case 0:  //Periodic
			xs = 0; xe = NX-1;
			xkind = DFT_R2C; 
			xkind_inv = DFT_C2R; 
			break;
		case 1:  //Dirichlet
			xs = 1; xe = NX-1;
			xkind = DST_1; 
			xkind_inv = DST_1; 
			break;
		case 2:  //Dirichlet - Neumann
			xs = 1; xe = NX;
			xkind = DST_3; 
			xkind_inv = DST_2; 
			break;
		case 3:  //Neumann
			xs = 0; xe = NX;
			xkind = DCT_1; 
			xkind_inv = DCT_1; 
			break;
		case 4:  //Neumann - Dirichlet
			xs = 0; xe = NX-1;
			xkind = DCT_3; 
			xkind_inv = DCT_2; 
			break;
		default:
			printf("bad boundary condition\n");
			exit(1);
			break;
	}

	switch (ybc) {
		case 0:  //Periodic
			ys = 0; ye = NY-1;
			ykind = DFT_R2C; 
			ykind_inv = DFT_C2R; 
			break;
		case 1:  //Dirichlet
			ys = 1; ye = NY-1;
			ykind = DST_1; 
			ykind_inv = DST_1; 
			break;
		case 2:  //Dirichlet - Neumann
			ys = 1; ye = NY;
			ykind = DST_3; 
			ykind_inv = DST_2; 
			break;
		case 3:  //Neumann
			ys = 0; ye = NY;
			ykind = DCT_1; 
			ykind_inv = DCT_1; 
			break;
		case 4:  //Neumann - Dirichlet
			ys = 0; ye = NY-1;
			ykind = DCT_3; 
			ykind_inv = DCT_2; 
			break;
		default:
			printf("bad boundary condition\n");
			exit(1);
			break;
	}

	switch (zbc) {
		case 0:  //Periodic
			zs = 0; ze = NZ-1;
			zkind = DFT_R2C; 
			zkind_inv = DFT_C2R; 
			break;
		case 1:  //Dirichlet
			zs = 1; ze = NZ-1;
			zkind = DST_1; 
			zkind_inv = DST_1; 
			break;
		case 2:  //Dirichlet - Neumann
			zs = 1; ze = NZ;
			zkind = DST_3; 
			zkind_inv = DST_2; 
			break;
		case 3:  //Neumann
			zs = 0; ze = NZ;
			zkind = DCT_1; 
			zkind_inv = DCT_1; 
			break;
		case 4:  //Neumann - Dirichlet
			zs = 0; ze = NZ-1;
			zkind = DCT_3; 
			zkind_inv = DCT_2; 
			break;
		default:
			printf("bad boundary condition\n");
			exit(1);
			break;
	}

	int nxp1 = NX + 1;
	int nyp1 = NY + 1;
	int nzp1 = NZ + 1;

	int yfstride = NX + 1; // ystride of array f;
	int zfstride = (NX + 1)*(NY + 1); // zstride of array f;

	int ystride = x_alloc_size;
	int zstride = x_alloc_size * y_alloc_size;

	_gp.nx = NX;
	_gp.ny = NY;
	_gp.nz = NZ;
	_gp.nxp1 = nxp1;
	_gp.nyp1 = nyp1;
	_gp.nzp1 = nzp1;
	_gp.xstride = 1;
	_gp.ystride = ystride;
	_gp.zstride = zstride;
	_gp.xfstride = 1;
	_gp.yfstride = yfstride;
	_gp.zfstride = zfstride;
	_gp.xs = xs;
	_gp.xe = xe;
	_gp.ys = ys;
	_gp.ye = ye;
	_gp.zs = zs;
	_gp.ze = ze;
	_gp.x_alloc_size = x_alloc_size;
	_gp.y_alloc_size = y_alloc_size;
	_gp.z_alloc_size = z_alloc_size;
	_gp.dx = dx;
	_gp.dy = dy;
	_gp.dz = dz;
	_gp.dx2 = dx2;
	_gp.dy2 = dy2;
	_gp.dz2 = dz2;
	int ratio = NX/2 * NY/2 * NZ/2;
	if (xbc == 0) ratio = ratio * 2;
	if (ybc == 0) ratio = ratio * 2;
	if (zbc == 0) ratio = ratio * 2;
	_gp.ratio = 1./(double)ratio;
	_gp.shiftx = shiftx;
	_gp.shifty = shifty;
	_gp.shiftz = shiftz;
	_gp.pidivnx = pidivnx;
	_gp.pidivny = pidivny;
	_gp.pidivnz = pidivnz;

	int dbx = 32;
	int dby = 8;
	int dbz = 4;

	cudaMemcpyToSymbol(_gp_d, &_gp, sizeof(_gp));


	dim3 block2d(dbx,dby);
	dim3 grid2d;
	grid2d = makegrid(dbx, dby, nyp1, nzp1);

	switch (xbc) {
		case 0: //P
			break;
		case 1: //D
			_3d_bc1_ini_x_<<< grid2d, block2d >>> (f);
			break;
		case 2: //D - N
			_3d_bc2_ini_x_<<< grid2d, block2d >>> (f, bcr);
			break;
		case 3: //N
			_3d_bc3_ini_x_<<< grid2d, block2d >>> (f, bcl, bcr);
			break;
		case 4: //N - D
			_3d_bc4_ini_x_<<< grid2d, block2d >>> (f, bcl);
			break;
	}

	grid2d = makegrid(dbx, dby, nxp1, nzp1);
	switch (ybc) {
		case 0: //P
			break;
		case 1: //D
			_3d_bc1_ini_y_<<< grid2d, block2d >>> (f);
			break;
		case 2: //D - N
			_3d_bc2_ini_y_<<< grid2d, block2d >>> (f, bct);
			break;
		case 3: //N
			_3d_bc3_ini_y_<<< grid2d, block2d >>> (f, bcb, bct);
			break;
		case 4: //N - D
			_3d_bc4_ini_y_<<< grid2d, block2d >>> (f, bcb);
			break;
	}

	grid2d = makegrid(dbx, dby, nxp1, nyp1);
	switch (zbc) {
		case 0: //P
			break;
		case 1: //D
			_3d_bc1_ini_z_<<< grid2d, block2d >>> (f);
			break;
		case 2: //D - N
			_3d_bc2_ini_z_<<< grid2d, block2d >>> (f, bce);
			break;
		case 3: //N
			_3d_bc3_ini_z_<<< grid2d, block2d >>> (f, bcf, bce);
			break;
		case 4: //N - D
			_3d_bc4_ini_z_<<< grid2d, block2d >>> (f, bcf);
			break;
	}

	int alloc_size = x_alloc_size * y_alloc_size * z_alloc_size;
	double * ex;
	double * ex_out;
	bool inplace = false;
	if (NX == NY && NX == NZ)
		//inplace = true;
		inplace = true;
	cudaMalloc(&ex, sizeof(double) * alloc_size);
	if (!inplace)
		cudaMalloc(&ex_out, sizeof(double) * alloc_size);
	else
		ex_out = ex;

	cudaMemset(ex, 0, sizeof(double) * alloc_size);
	check_error();

	// initialize ex

	dim3 block3d(dbx, dby, dbz);
	dim3 grid3d;

	grid3d = makegrid(dbx, dby, dbz, nxp1, nyp1, nzp1);
	_ini_ex_ <<< grid3d, block3d >>> (ex, f);
	check_error();

	// solve:
	do_transform(ex, ex_out, NX, NY, NZ, xkind, ykind, zkind);
	//check_error();

	// we need to deal with periodic condition in special way 
	
	if (xbc == 0) x_alloc_size /= 2;
	if (ybc == 0) y_alloc_size /= 2;
	if (zbc == 0) z_alloc_size /= 2;
	grid3d = makegrid(dbx, dby, dbz, x_alloc_size, y_alloc_size, z_alloc_size);
	int periodic = 0;
	if (!xbc || !ybc || !zbc) periodic = 1;

	//_nonpoisson_devide_19points_ <<< grid3d, block3d >>> (ex, lambda);
	//_nonpoisson_devide_ <<< grid3d, block3d >>> (ex, lambda);

	bool needperturb = false;
	if (lambda == 0. && (xbc==0 ||xbc==3) && (ybc==0 ||ybc==3) && (zbc==0 ||zbc==3))
		needperturb = true;

	if (periodic)
	{
				_periodic_divide_ <<< grid3d, block3d >>> (ex_out, lambda, needperturb, xbc, ybc, zbc);
				check_error();
	}
	else{
				//_nonpoisson_devide_19points_ <<< grid3d, block3d >>> (ex, lambda);
				_divide_ <<< grid3d, block3d >>> (ex_out, lambda, needperturb);
				check_error();
	}


	do_transform(ex_out, ex, NX, NY, NZ, xkind_inv, ykind_inv, zkind_inv);
	//check_error();

	if (!inplace)
		cudaFree(ex_out);
	// put solution in original array

	grid3d = makegrid(dbx, dby, dbz, nxp1, nyp1, nzp1);
	_store_f_ <<< grid3d, block3d >>> (ex, f);
	check_error();

	// periodic bc:

	if (xbc ==0)
	{
		grid2d = makegrid(dbx, dby, nyp1, nzp1);
		_3d_bc0_store_x_ <<< grid2d, block2d >>> (f);
	}
	if (ybc ==0)
	{
		grid2d = makegrid(dbx, dby, nxp1, nzp1);
		_3d_bc0_store_y_ <<< grid2d, block2d >>> (f);
	}
	if (zbc ==0)
	{
		grid2d = makegrid(dbx, dby, nxp1, nyp1);
		_3d_bc0_store_z_ <<< grid2d, block2d >>> (f);
	}

	if (needperturb)
	{
//		printf("perturb is : %.16f\n", perturb/(TX*TY*TZ));
	}

	cudaFree(ex);
	return 0;
}
