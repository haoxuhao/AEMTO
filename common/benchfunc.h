#ifndef __BENCHFUNC_H__
#define __BENCHFUNC_H__
#include "config.h"

#ifndef M_E
#define M_E  2.7182818284590452353602874713526625
#endif
#ifndef PI
#define PI 3.1415926535897932384626433832795029
#endif
namespace benchfunc {

Real sphere(vector<Real> &z) /* Sphere */
{
	Real f = 0.0;
	int nx = z.size();	
	Real *ze = z.data();

	for (int i = 0; i<nx; i++)
	{
		f += ze[i] * ze[i];
	}
	return f;
};

Real rosenbrock(vector<Real> &z) /* Rosenbrock's */
{
	int i;
	Real tmp1, tmp2;
	Real f = 0.0;
	Real *ze = z.data();
	int nx = z.size();

	for (i = 0; i < nx - 1; i++)
	{
		tmp1 = ze[i] * ze[i] - ze[i + 1];
		tmp2 = ze[i] - 1.0;
		f += 100.0*tmp1*tmp1 + tmp2*tmp2;
	}
	return f;
};

Real ackley(vector<Real> &z) /* Ackley's  */
{
	int i;
	Real sum1, sum2;
	sum1 = 0.0;
	sum2 = 0.0;
	Real *ze = z.data();
	Real f = 0;
	int nx = z.size();

	for (i = 0; i<nx; i++)
	{
		sum1 += ze[i] * ze[i];
		sum2 += cos(2.0*PI*ze[i]);
	}
	sum1 = -0.2*sqrt(sum1 / (Real)nx);
	sum2 /= (Real)nx;
	f = M_E - 20.0*exp(sum1) + 20.0 - exp(sum2);
	return f;
};

Real weierstrass(vector<Real> &z) /* Weierstrass's  */
{
	int i, j, k_max;
	Real f = 0;
	Real sum, sum2, a, b;
	a = 0.5;
	b = 3.0;
	k_max = 20;
	f = 0.0;
	Real *ze = z.data();
	int nx = z.size();

	static vector<Real> pow_a_values_weierstrass;
	static vector<Real> pow_b_values_weierstrass;
	static Real sum2_weierstrass = 0;
	static bool one_run = true;
	if (one_run) {
		Real pw_a, pw_b;
		for(int j = 0; j <= k_max; j++)	{
			pw_a = pow(a, j);
			pw_b = pow(b, j);
			pow_a_values_weierstrass.push_back(pw_a);
			pow_b_values_weierstrass.push_back(pw_b);
			sum2_weierstrass += pw_a * cos(2 * PI * pw_b * 0.5);
		}
		one_run = false;
	}

	Real twopi = 2.0 * PI;
	for(i = 0; i < nx; i++)
	{
		sum = 0;
		for (j = 0; j <= k_max; j++)
		{
			sum += pow_a_values_weierstrass[j] * \
				cos(twopi * pow_b_values_weierstrass[j] * (ze[i] + 0.5));
		}
		f += sum;
	}
	f -= nx*sum2_weierstrass;
	return f;
};


Real griewank(vector<Real> &z) /* Griewank's  */
{
	int i;
	Real f = 0.0;
	int nx = z.size();
	Real s, p;
	s = 0.0;
	p = 1.0;
	Real *ze = z.data();

	for (i = 0; i<nx; i++)
	{
		s += ze[i] * ze[i];
		p *= cos(ze[i] / sqrt(1.0 + i));
	}
	f = 1.0 + s / 4000.0 - p;
	return f;
};

Real rastrigin(vector<Real> &z) /* Rastrigin's  */
{
	int i;
	Real f = 0.0;
	Real *ze = z.data();
	int nx = z.size();
	for (i = 0; i<nx; i++)
	{
		f += (ze[i] * ze[i] - 10.0*cos(2.0*PI*ze[i]) + 10.0);
	}
	return f;
};

Real schwefel(vector<Real> &z) /* Schwefel's  */
{
	int i;
	Real tmp;
	Real f = 0.0;
	int nx = z.size();
	Real *ze = z.data(); 

	for (i = 0; i<nx; i++)
	{
		if (ze[i]>500)
		{
			f -= (500.0 - fmod(ze[i], 500))*sin(pow(500.0 - fmod(ze[i], 500), 0.5));
			tmp = (ze[i] - 500.0) / 100;
			f += tmp*tmp / nx;
		}
		else if (ze[i]<-500)
		{
			f -= (-500.0 + fmod(fabs(ze[i]), 500))*sin(pow(500.0 - fmod(fabs(ze[i]), 500), 0.5));
			tmp = (ze[i] + 500.0) / 100;
			f += tmp*tmp / nx;
		}
		else
			f -= ze[i] * sin(pow(fabs(ze[i]), 0.5));
	}
	f += 4.189828872724338e+002*nx;
	return f;
};
};
#endif