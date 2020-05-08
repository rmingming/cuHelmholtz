struct par_constant {
	int nx;
	int ny;
	int nz;
	int nxp1;
	int nyp1;
	int nzp1;
	int xstride;
	int ystride;
	int zstride;
	int xfstride;
	int yfstride;
	int zfstride;
	int xs, xe;
	int ys, ye;
	int zs, ze;
	int x_alloc_size;
	int y_alloc_size;
	int z_alloc_size;
	double dx;
	double dy;
	double dz;
	double dx2;
	double dy2;
	double dz2;
	double ratio; // 1.0/ (NX/2 * NY/2 * NZ/2)
	double shiftx;
	double shifty;
	double shiftz;
	double pidivnx;
	double pidivny;
	double pidivnz;
} ;

__constant__ __align__(16) par_constant _gp_d;
struct par_constant _gp;

#define let_2d_idx(i,j) \
	int i = blockDim.x * blockIdx.x + threadIdx.x; \
	int j = blockDim.y * blockIdx.y + threadIdx.y; \

#define let_3d_idx(i,j,k) \
	int i = blockDim.x * blockIdx.x + threadIdx.x; \
	int j = blockDim.y * blockIdx.y + threadIdx.y; \
	int k = blockDim.z * blockIdx.z + threadIdx.z; 
/*
#define let_2d_idx(i,j) \
	int i = (blockIdx.x << 5) + threadIdx.x; \
	int j = (blockIdx.y << 3) + threadIdx.y; \

#define let_3d_idx(i,j,k) \
	int i = (blockIdx.x << 5) + threadIdx.x; \
	int j = (blockIdx.y << 3) + threadIdx.y; \
	int k = (blockIdx.z << 2) + threadIdx.z; 
*/

// x direction boundary
__global__
void _3d_bc0_store_x_ (double * f)
{
	let_2d_idx(j,k);
	//if ((k<_gp_d.zs)||(k>_gp_d.ze)) return;
	//if ((j<_gp_d.ys)||(j>_gp_d.ye)) return;
	if ((k<0)||(k>_gp_d.nz)) return;
	if ((j<0)||(j>_gp_d.ny)) return;
	f[k*_gp_d.zfstride+j*_gp_d.yfstride+_gp_d.nx]
		= f[k*_gp_d.zfstride+j*_gp_d.yfstride]; 
}

__global__
void _3d_bc1_ini_x_ (double * f)
{
	let_2d_idx(j,k);
	if ((k<_gp_d.zs)||(k>_gp_d.ze)) return;
	if ((j<_gp_d.ys)||(j>_gp_d.ye)) return;
	f[k*_gp_d.zfstride+j*_gp_d.yfstride+_gp_d.xs]
		-= f[k*_gp_d.zfstride+j*_gp_d.yfstride] * _gp_d.dx2;
	f[k*_gp_d.zfstride+j*_gp_d.yfstride+_gp_d.xe]
		-= f[k*_gp_d.zfstride+j*_gp_d.yfstride+_gp_d.nx] * _gp_d.dx2;
}
__global__
void _3d_bc2_ini_x_ (double * f, double *bcr)
{
	let_2d_idx(j,k);
	if ((k<_gp_d.zs)||(k>_gp_d.ze)) return;
	if ((j<_gp_d.ys)||(j>_gp_d.ye)) return;
	f[k*_gp_d.zfstride+j*_gp_d.yfstride+_gp_d.xs]
		-= f[k*_gp_d.zfstride+j*_gp_d.yfstride] * _gp_d.dx2;
	f[k*_gp_d.zfstride+j*_gp_d.yfstride+_gp_d.xe]
		-= 2 * bcr[k*_gp_d.nyp1+j] * _gp_d.dx;
}
__global__
void _3d_bc3_ini_x_ (double * f, double *bcl, double *bcr)
{
	let_2d_idx(j,k);
	if ((k<_gp_d.zs)||(k>_gp_d.ze)) return;
	if ((j<_gp_d.ys)||(j>_gp_d.ye)) return;
	f[k*_gp_d.zfstride+j*_gp_d.yfstride+_gp_d.xs]
		+= 2 * bcl[k*_gp_d.nyp1+j] * _gp_d.dx;
	f[k*_gp_d.zfstride+j*_gp_d.yfstride+_gp_d.xe]
		-= 2 * bcr[k*_gp_d.nyp1+j] * _gp_d.dx;
}
__global__
void _3d_bc4_ini_x_ (double * f, double *bcl)
{
	let_2d_idx(j,k);
	if ((k<_gp_d.zs)||(k>_gp_d.ze)) return;
	if ((j<_gp_d.ys)||(j>_gp_d.ye)) return;
	f[k*_gp_d.zfstride+j*_gp_d.yfstride+_gp_d.xs]
		+= 2 * bcl[k*_gp_d.nyp1+j] * _gp_d.dx;
	f[k*_gp_d.zfstride+j*_gp_d.yfstride+_gp_d.xe]
		-= f[k*_gp_d.zfstride+j*_gp_d.yfstride+_gp_d.nx] * _gp_d.dx2;
}

// y direction boundary
__global__
void _3d_bc0_store_y_ (double * f)
{
	let_2d_idx(i,k);
	//if ((k<_gp_d.zs)||(k>_gp_d.ze)) return;
	//if ((i<_gp_d.xs)||(i>_gp_d.xe)) return;
	if ((k<0)||(k>_gp_d.nz)) return;
	if ((i<0)||(i>_gp_d.nx)) return;
	f[k*_gp_d.zfstride+_gp_d.ny*_gp_d.yfstride+i]
		= f[k*_gp_d.zfstride+i]; 
}
__global__
void _3d_bc1_ini_y_ (double * f)
{
	let_2d_idx(i,k);
	if ((k<_gp_d.zs)||(k>_gp_d.ze)) return;
	if ((i<_gp_d.xs)||(i>_gp_d.xe)) return;
	f[k*_gp_d.zfstride+_gp_d.ys*_gp_d.yfstride+i]
		-= f[k*_gp_d.zfstride+i] * _gp_d.dy2;
	f[k*_gp_d.zfstride+_gp_d.ye*_gp_d.yfstride+i]
		-= f[k*_gp_d.zfstride+_gp_d.ny*_gp_d.yfstride+i] * _gp_d.dy2;
}
__global__
void _3d_bc2_ini_y_ (double * f, double *bct)
{
	let_2d_idx(i,k);
	if ((k<_gp_d.zs)||(k>_gp_d.ze)) return;
	if ((i<_gp_d.xs)||(i>_gp_d.xe)) return;
	f[k*_gp_d.zfstride+_gp_d.ys*_gp_d.yfstride+i]
		-= f[k*_gp_d.zfstride+i] * _gp_d.dy2;
	f[k*_gp_d.zfstride+_gp_d.ye*_gp_d.yfstride+i]
		-= 2 * bct[k*_gp_d.nxp1+i] * _gp_d.dy;
}
__global__
void _3d_bc3_ini_y_ (double * f, double *bcb, double *bct)
{
	let_2d_idx(i,k);
	if ((k<_gp_d.zs)||(k>_gp_d.ze)) return;
	if ((i<_gp_d.xs)||(i>_gp_d.xe)) return;
	f[k*_gp_d.zfstride+_gp_d.ys*_gp_d.yfstride+i]
		+= 2 * bcb[k*_gp_d.nxp1+i] * _gp_d.dy;
	f[k*_gp_d.zfstride+_gp_d.ye*_gp_d.yfstride+i]
		-= 2 * bct[k*_gp_d.nxp1+i] * _gp_d.dy;
}
__global__
void _3d_bc4_ini_y_ (double * f, double *bcb)
{
	let_2d_idx(i,k);
	if ((k<_gp_d.zs)||(k>_gp_d.ze)) return;
	if ((i<_gp_d.xs)||(i>_gp_d.xe)) return;
	f[k*_gp_d.zfstride+_gp_d.ys*_gp_d.yfstride+i]
		+= 2 * bcb[k*_gp_d.nxp1+i] * _gp_d.dy;
	f[k*_gp_d.zfstride+_gp_d.ye*_gp_d.yfstride+i]
		-= f[k*_gp_d.zfstride+_gp_d.ny*_gp_d.yfstride+i] * _gp_d.dy2;
}

// z direction boundary
__global__
void _3d_bc0_store_z_ (double * f)
{
	let_2d_idx(i,j);
	//if ((j<_gp_d.ys)||(j>_gp_d.ye)) return;
	//if ((i<_gp_d.xs)||(i>_gp_d.xe)) return;
	if ((j<0)||(j>_gp_d.ny)) return;
	if ((i<0)||(i>_gp_d.nx)) return;
	f[_gp_d.nz*_gp_d.zfstride+j*_gp_d.yfstride+i]
		= f[j*_gp_d.yfstride+i];
}
__global__
void _3d_bc1_ini_z_ (double * f)
{
	let_2d_idx(i,j);
	if ((j<_gp_d.ys)||(j>_gp_d.ye)) return;
	if ((i<_gp_d.xs)||(i>_gp_d.xe)) return;
	f[_gp_d.zs*_gp_d.zfstride+j*_gp_d.yfstride+i]
		-= f[j*_gp_d.yfstride+i] * _gp_d.dz2;
	f[_gp_d.ze*_gp_d.zfstride+j*_gp_d.yfstride+i]
		-= f[_gp_d.nz*_gp_d.zfstride+j*_gp_d.yfstride+i] * _gp_d.dz2;
}
__global__
void _3d_bc2_ini_z_ (double * f, double *bce)
{
	let_2d_idx(i,j);
	if ((j<_gp_d.ys)||(j>_gp_d.ye)) return;
	if ((i<_gp_d.xs)||(i>_gp_d.xe)) return;
	f[_gp_d.zs*_gp_d.zfstride+j*_gp_d.yfstride+i]
		-= f[j*_gp_d.yfstride+i] * _gp_d.dz2;
	f[_gp_d.ze*_gp_d.zfstride+j*_gp_d.yfstride+i]
		-= 2 * bce[j*_gp_d.nxp1+i] * _gp_d.dz;
}
__global__
void _3d_bc3_ini_z_ (double * f, double *bcf, double *bce)
{
	let_2d_idx(i,j);
	if ((j<_gp_d.ys)||(j>_gp_d.ye)) return;
	if ((i<_gp_d.xs)||(i>_gp_d.xe)) return;
	f[_gp_d.zs*_gp_d.zfstride+j*_gp_d.yfstride+i]
		+= 2 * bcf[j*_gp_d.nxp1+i] * _gp_d.dz;
	f[_gp_d.ze*_gp_d.zfstride+j*_gp_d.yfstride+i]
		-= 2 * bce[j*_gp_d.nxp1+i] * _gp_d.dz;
}
__global__
void _3d_bc4_ini_z_ (double * f, double *bcf)
{
	let_2d_idx(i,j);
	if ((j<_gp_d.ys)||(j>_gp_d.ye)) return;
	if ((i<_gp_d.xs)||(i>_gp_d.xe)) return;
	f[_gp_d.zs*_gp_d.zfstride+j*_gp_d.yfstride+i]
		+= 2 * bcf[j*_gp_d.nxp1+i] * _gp_d.dz;
	f[_gp_d.ze*_gp_d.zfstride+j*_gp_d.yfstride+i]
		-= f[_gp_d.nz*_gp_d.zfstride+j*_gp_d.yfstride+i] * _gp_d.dz2;
}

//initialize ex
__global__
void _ini_ex_(double *ex, double *f)
{
	let_3d_idx(i,j,k);
	if ((k<_gp_d.zs)||(k>_gp_d.ze)) return;
	if ((j<_gp_d.ys)||(j>_gp_d.ye)) return;
	if ((i<_gp_d.xs)||(i>_gp_d.xe)) return;
	ex[k*_gp_d.zstride + j*_gp_d.ystride + i]
		= f[k*_gp_d.zfstride + j*_gp_d.yfstride + i];
}

// store f
__global__
void _store_f_(double *ex, double *f)
{
	let_3d_idx(i,j,k);
	if ((k<_gp_d.zs)||(k>_gp_d.ze)) return;
	if ((j<_gp_d.ys)||(j>_gp_d.ye)) return;
	if ((i<_gp_d.xs)||(i>_gp_d.xe)) return;
	f[k*_gp_d.zfstride + j*_gp_d.yfstride + i] 
		= ex[k*_gp_d.zstride + j*_gp_d.ystride + i] * _gp_d.ratio;
}

#define PI 3.14159265358979323846

__global__
void _nonpoisson_devide_19points_(double * ex, double lambda)
{
	let_3d_idx(i,j,k);
	if ((k<_gp_d.zs)||(k>_gp_d.ze)) return;
	if ((j<_gp_d.ys)||(j>_gp_d.ye)) return;
	if ((i<_gp_d.xs)||(i>_gp_d.xe)) return;

	double cosi = cos(i * _gp_d.pidivnx);
	double cosj = cos(j * _gp_d.pidivny);
	double cosk = cos(k * _gp_d.pidivnz);
	double coef = 2./3. * (cosi * cosj + cosi * cosk + cosj * cosk
			+ cosi + cosj + cosk) - 4;
	coef *= _gp_d.dx2;
	coef += lambda;
	
	double fijk = ex[k*_gp_d.zstride+j*_gp_d.ystride+i] * (
			0.5 + 1./6. * (cosi + cosj + cosk) );
	ex[k*_gp_d.zstride+j*_gp_d.ystride+i] = fijk/coef;
}

__global__
void _periodic_divide_(double * ex, double lambda, bool needperturb, int xbc, int ybc, int zbc)
{
	let_3d_idx(i,j,k);
	if (xbc==0) 
	{
		if (i>_gp_d.nx/2) return;
	}else{
		if ((i<_gp_d.xs)||(i>_gp_d.xe)) return;
	}
	if (ybc==0) 
	{
		if (j>_gp_d.ny/2) return;
	}else{
		if ((j<_gp_d.ys)||(j>_gp_d.ye)) return;
	}
	if (zbc==0) 
	{
		if (k>_gp_d.nz/2) return;
	}else{
		if ((k<_gp_d.zs)||(k>_gp_d.ze)) return;
	}

	double eig;

	if (needperturb && i==0 && j==0 && k==0)
	{
		printf("encounter needperturb case\n");
		eig = 0.;
	}
	else 
		eig = 1./ ( 
				(2*cos(i*_gp_d.pidivnx + _gp_d.shiftx)-2)*_gp_d.dx2 
				+ (2*cos(j*_gp_d.pidivny + _gp_d.shifty)-2)*_gp_d.dy2 
				+ (2*cos(k*_gp_d.pidivnz + _gp_d.shiftz)-2)*_gp_d.dz2 
				+ lambda
				);

	if (xbc == 0 && ybc == 0 && zbc == 0)
	{
		int base = (2*k)*_gp_d.zstride+(2*j)*_gp_d.ystride+2*i;
		ex[base] *= eig;
		ex[base + 1] *= eig;
		ex[base + _gp_d.ystride] *= eig;
		ex[base + _gp_d.ystride + 1] *= eig;
		ex[base + _gp_d.zstride] *= eig;
		ex[base + _gp_d.zstride + 1] *= eig;
		ex[base + _gp_d.zstride + _gp_d.ystride] *= eig;
		ex[base + _gp_d.zstride + _gp_d.ystride + 1] *= eig;
		/*
		ex[(2*k)*_gp_d.zstride+(2*j)*_gp_d.ystride+2*i] *= eig;
		ex[(2*k)*_gp_d.zstride+(2*j)*_gp_d.ystride+2*i+1] *= eig;
		ex[(2*k)*_gp_d.zstride+(2*j+1)*_gp_d.ystride+2*i] *= eig;
		ex[(2*k)*_gp_d.zstride+(2*j+1)*_gp_d.ystride+2*i+1] *= eig;
		ex[(2*k+1)*_gp_d.zstride+(2*j)*_gp_d.ystride+2*i] *= eig;
		ex[(2*k+1)*_gp_d.zstride+(2*j)*_gp_d.ystride+2*i+1] *= eig;
		ex[(2*k+1)*_gp_d.zstride+(2*j+1)*_gp_d.ystride+2*i] *= eig;
		ex[(2*k+1)*_gp_d.zstride+(2*j+1)*_gp_d.ystride+2*i+1] *= eig;
		*/
		return;
	}
	if (xbc == 0 && ybc == 0)
	{
		int base = k*_gp_d.zstride+(2*j)*_gp_d.ystride+2*i;
		ex[base] *= eig;
		ex[base + 1] *= eig;
		ex[base + _gp_d.ystride] *= eig;
		ex[base + _gp_d.ystride + 1] *= eig;
		/*
		ex[k*_gp_d.zstride+(2*j)*_gp_d.ystride+2*i] *= eig;
		ex[k*_gp_d.zstride+(2*j)*_gp_d.ystride+2*i+1] *= eig;
		ex[k*_gp_d.zstride+(2*j+1)*_gp_d.ystride+2*i] *= eig;
		ex[k*_gp_d.zstride+(2*j+1)*_gp_d.ystride+2*i+1] *= eig;
		*/
		return;
	}
	if (xbc == 0 && zbc == 0)
	{
		int base = 2*k*_gp_d.zstride+j*_gp_d.ystride+2*i;
		ex[base] *= eig;
		ex[base + 1] *= eig;
		ex[base + _gp_d.zstride] *= eig;
		ex[base + _gp_d.zstride + 1] *= eig;
		return;
	}
	if (ybc == 0 && zbc == 0)
	{
		int base = 2*k*_gp_d.zstride+2*j*_gp_d.ystride+i;
		ex[base] *= eig;
		ex[base + _gp_d.ystride] *= eig;
		ex[base + _gp_d.zstride] *= eig;
		ex[base + _gp_d.zstride + _gp_d.ystride] *= eig;
		return;
	}

	if (xbc == 0)
	{
		int base = k*_gp_d.zstride+j*_gp_d.ystride+2*i;
		ex[base] *= eig;
		ex[base + 1] *= eig;
		return;
	}
	if (ybc == 0)
	{
		int base = k*_gp_d.zstride+2*j*_gp_d.ystride+i;
		ex[base] *= eig;
		ex[base + _gp_d.ystride] *= eig;
		return;
	}
	if (zbc == 0)
	{
		int base = 2*k*_gp_d.zstride+j*_gp_d.ystride+i;
		ex[base] *= eig;
		ex[base + _gp_d.zstride] *= eig;
		return;
	}
}

__global__
void _divide_(double * ex, double lambda, bool needperturb)
{
	let_3d_idx(i,j,k);
	if ((k<_gp_d.zs)||(k>_gp_d.ze)) return;
	if ((j<_gp_d.ys)||(j>_gp_d.ye)) return;
	if ((i<_gp_d.xs)||(i>_gp_d.xe)) return;
	//if ((k>_gp_d.ze)) return;
	//if ((j>_gp_d.ye)) return;
	//if ((i>_gp_d.xe)) return;

	double eig;

	if (needperturb && i==0 && j==0 && k==0)
	{
		printf("encounter needperturb case\n");
		eig = 0.;
	}
	else 
		eig = 1./ ( 
				(2.*cos(i*_gp_d.pidivnx + _gp_d.shiftx)-2.)*_gp_d.dx2 
				+ (2.*cos(j*_gp_d.pidivny + _gp_d.shifty)-2.)*_gp_d.dy2 
				+ (2.*cos(k*_gp_d.pidivnz + _gp_d.shiftz)-2.)*_gp_d.dz2 
				+ lambda
				);

	ex[k*_gp_d.zstride+j*_gp_d.ystride+i] *= eig;
}

