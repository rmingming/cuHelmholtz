#pragma once
#include <cmath>
#define PI 3.14159265358979323846

/*
 * Attention: 
 * If you want to use the periodic boundary condition,
 * you must make sure the following functions return 
 * identical values at two ends in the periodic direction
 */

// case 1: u = 1/3 * sin(2*PI*x) * sin(2*PI*y) * sin(2*PI*z);
inline double ufunc(double x, double y, double z)
{
	return 1./3.*sin(2*PI*x)*sin(2*PI*y)*sin(2*PI*z);
}
inline double uxfunc(double x, double y, double z)
{
	return 2.*PI/3.*cos(2*PI*x)*sin(2*PI*y)*sin(2*PI*z);
}
inline double uyfunc(double x, double y, double z)
{
	return 2.*PI/3.*sin(2*PI*x)*cos(2*PI*y)*sin(2*PI*z);
}
inline double uzfunc(double x, double y, double z)
{
	return 2.*PI/3.*sin(2*PI*x)*sin(2*PI*y)*cos(2*PI*z);
}
inline double laplaceufunc(double x, double y, double z)
{
	return -4.*PI*PI*sin(2*PI*x)*sin(2*PI*y)*sin(2*PI*z);
}

/*
// case 2: u = e^x + x*y*z;
inline double ufunc(double x, double y, double z)
{
	return exp(x)+x*y*z;
}
inline double uxfunc(double x, double y, double z)
{
	return exp(x)+y*z;
}
inline double uyfunc(double x, double y, double z)
{
	return x*z;
}
inline double uzfunc(double x, double y, double z)
{
	return x*y;
}
inline double laplaceufunc(double x, double y, double z)
{
	return exp(x);
}
*/

/*
// case 3: u = sin(2*PI*x) + y*z;
inline double ufunc(double x, double y, double z)
{
	return sin(2*PI*x)+y*z;
}
inline double uxfunc(double x, double y, double z)
{
	return 2*PI*cos(2*PI*x);
}
inline double uyfunc(double x, double y, double z)
{
	return z;
}
inline double uzfunc(double x, double y, double z)
{
	return y;
}
inline double laplaceufunc(double x, double y, double z)
{
	return -4.*PI*PI*sin(2*PI*x);
}
*/
