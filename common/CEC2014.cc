#include "CEC2014.h"



CEC2014::CEC2014() : sum2_weierstrass(0)
{
	// Pre computed Weierstass constants
	real a = 0.5;
	real b = 3.0;
	real k_max = 20;
	real pw_a, pw_b;
	
	for(int j = 0; j <= k_max; j++)
	{
		pw_a = pow(a, j);
		pw_b = pow(b, j);
		pow_a_values_weierstrass.push_back(pw_a);
		pow_b_values_weierstrass.push_back(pw_b);
		sum2_weierstrass += pw_a * cos(2 * PI * pw_b * 0.5);
	}
}

CEC2014::~CEC2014()
{
}
real CEC2014::EvaluateFitness(real * elements)
{
	real fitness_value = 0;
	switch (func_id_)
	{
	case 1:
		ellips_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 1);
		fitness_value += 100.0;
		break;
	case 2:
		bent_cigar_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 1);
		fitness_value += 200.0;
		break;
	case 3:
		discus_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 1);
		fitness_value += 300.0;
		break;
	case 4:
		rosenbrock_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 1, 1, 2.048 / (real)100);
		fitness_value += 400.0;
		break;
	case 5:
		ackley_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 1);
		fitness_value += 500.0;
		break;
	case 6:
		weierstrass_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 0.5 / (real)100);
		fitness_value += 600.0;
		break;
	case 7:
		griewank_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 6);
		fitness_value += 700.0;
		break;
	case 8:
		rastrigin_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 0, 0, 5.12 / (real)100);
		fitness_value += 800.0;
		break;
	case 9:
		rastrigin_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 5.12 / (real)100);
		fitness_value += 900.0;
		break;
	case 10:
		schwefel_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 0, 0, 10);
		fitness_value += 1000.0;
		break;
	case 11:
		schwefel_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 10);
		fitness_value += 1100.0;
		break;
	case 12:
		katsuura_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 5 / (real)100);
		fitness_value += 1200.0;
		break;
	case 13:
		happycat_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 5 / (real)100);
		fitness_value += 1300.0;
		break;
	case 14:
		hgbat_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 5 / (real)100);
		fitness_value += 1400.0;
		break;
	case 15:
		grie_rosen_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 1, 1, 5 / (real)100);
		fitness_value += 1500.0;
		break;
	case 16:
		escaffer6_func(elements, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 1);
		fitness_value += 1600.0;
		break;
	case 17:
		hf01(elements, &fitness_value, dim_, OShift, Mdata, SS, 1, 1);
		fitness_value += 1700.0;
		break;
	case 18:
		hf02(elements, &fitness_value, dim_, OShift, Mdata, SS, 1, 1);
		fitness_value += 1800.0;
		break;
	case 19:
		hf03(elements, &fitness_value, dim_, OShift, Mdata, SS, 1, 1);
		fitness_value += 1900.0;
		break;
	case 20:
		hf04(elements, &fitness_value, dim_, OShift, Mdata, SS, 1, 1);
		fitness_value += 2000.0;
		break;
	case 21:
		hf05(elements, &fitness_value, dim_, OShift, Mdata, SS, 1, 1);
		fitness_value += 2100.0;
		break;
	case 22:
		hf06(elements, &fitness_value, dim_, OShift, Mdata, SS, 1, 1);
		fitness_value += 2200.0;
		break;
	case 23:
		cf01(elements, &fitness_value, dim_, OShift, Mdata, 1);
		fitness_value += 2300.0;
		break;
	case 24:
		cf02(elements, &fitness_value, dim_, OShift, Mdata, 1);
		fitness_value += 2400.0;
		break;
	case 25:
		cf03(elements, &fitness_value, dim_, OShift, Mdata, 1);
		fitness_value += 2500.0;
		break;
	case 26:
		cf04(elements, &fitness_value, dim_, OShift, Mdata, 1);
		fitness_value += 2600.0;
		break;
	case 27:
		cf05(elements, &fitness_value, dim_, OShift, Mdata, 1);
		fitness_value += 2700.0;
		break;
	case 28:
		cf06(elements, &fitness_value, dim_, OShift, Mdata, 1);
		fitness_value += 2800.0;
		break;
	case 29:
		cf07(elements, &fitness_value, dim_, OShift, Mdata, SS, 1);
		fitness_value += 2900.0;
		break;
	case 30:
		cf08(elements, &fitness_value, dim_, OShift, Mdata, SS, 1);
		fitness_value += 3000.0;
		break;
	default:
		printf("\nError: There are only 30 test functions in this test suite!\n");
		fitness_value = 0.0;
		break;
	}
	return fitness_value - 100 * func_id_;
}
real CEC2014::EvaluateFitness(const vector<real> & elements)
{
	real fitness_value = 0;
	copy(elements.begin(), elements.end(), pop_original_);

	switch (func_id_)
	{
	case 1:
		ellips_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 1);
		fitness_value += 100.0;
		break;
	case 2:
		bent_cigar_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 1);
		fitness_value += 200.0;
		break;
	case 3:
		discus_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 1);
		fitness_value += 300.0;
		break;
	case 4:
		rosenbrock_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 1, 1, 2.048 / (real)100);
		fitness_value += 400.0;
		break;
	case 5:
		ackley_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 1);
		fitness_value += 500.0;
		break;
	case 6:
		weierstrass_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 0.5 / (real)100);
		fitness_value += 600.0;
		break;
	case 7:
		griewank_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 6);
		fitness_value += 700.0;
		break;
	case 8:
		rastrigin_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 0, 0, 5.12 / (real)100);
		fitness_value += 800.0;
		break;
	case 9:
		rastrigin_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 5.12 / (real)100);
		fitness_value += 900.0;
		break;
	case 10:
		schwefel_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 0, 0, 10);
		fitness_value += 1000.0;
		break;
	case 11:
		schwefel_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 10);
		fitness_value += 1100.0;
		break;
	case 12:
		katsuura_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 5 / (real)100);
		fitness_value += 1200.0;
		break;
	case 13:
		happycat_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 5 / (real)100);
		fitness_value += 1300.0;
		break;
	case 14:
		hgbat_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 5 / (real)100);
		fitness_value += 1400.0;
		break;
	case 15:
		grie_rosen_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 1, 1, 5 / (real)100);
		fitness_value += 1500.0;
		break;
	case 16:
		escaffer6_func(pop_original_, &fitness_value, dim_, OShift, Mdata, 1, 1, 0, 1);
		fitness_value += 1600.0;
		break;
	case 17:
		hf01(pop_original_, &fitness_value, dim_, OShift, Mdata, SS, 1, 1);
		fitness_value += 1700.0;
		break;
	case 18:
		hf02(pop_original_, &fitness_value, dim_, OShift, Mdata, SS, 1, 1);
		fitness_value += 1800.0;
		break;
	case 19:
		hf03(pop_original_, &fitness_value, dim_, OShift, Mdata, SS, 1, 1);
		fitness_value += 1900.0;
		break;
	case 20:
		hf04(pop_original_, &fitness_value, dim_, OShift, Mdata, SS, 1, 1);
		fitness_value += 2000.0;
		break;
	case 21:
		hf05(pop_original_, &fitness_value, dim_, OShift, Mdata, SS, 1, 1);
		fitness_value += 2100.0;
		break;
	case 22:
		hf06(pop_original_, &fitness_value, dim_, OShift, Mdata, SS, 1, 1);
		fitness_value += 2200.0;
		break;
	case 23:
		cf01(pop_original_, &fitness_value, dim_, OShift, Mdata, 1);
		fitness_value += 2300.0;
		break;
	case 24:
		cf02(pop_original_, &fitness_value, dim_, OShift, Mdata, 1);
		fitness_value += 2400.0;
		break;
	case 25:
		cf03(pop_original_, &fitness_value, dim_, OShift, Mdata, 1);
		fitness_value += 2500.0;
		break;
	case 26:
		cf04(pop_original_, &fitness_value, dim_, OShift, Mdata, 1);
		fitness_value += 2600.0;
		break;
	case 27:
		cf05(pop_original_, &fitness_value, dim_, OShift, Mdata, 1);
		fitness_value += 2700.0;
		break;
	case 28:
		cf06(pop_original_, &fitness_value, dim_, OShift, Mdata, 1);
		fitness_value += 2800.0;
		break;
	case 29:
		cf07(pop_original_, &fitness_value, dim_, OShift, Mdata, SS, 1);
		fitness_value += 2900.0;
		break;
	case 30:
		cf08(pop_original_, &fitness_value, dim_, OShift, Mdata, SS, 1);
		fitness_value += 3000.0;
		break;
	default:
		printf("\nError: There are only 30 test functions in this test suite!\n");
		fitness_value = 0.0;
		break;
	}
	return fitness_value - 100 * func_id_;
}


int	CEC2014::Uninitialize()
{
	delete[]pop_original_;
	delete[]OShift;
	delete[]Mdata;
	delete[]ye;
	delete[]ze;
	delete[]x_bound;
	if ((func_id_ >= 17 && func_id_ <= 22) || func_id_ == 29 || func_id_ == 30)
		delete[]SS;
	return 0;
}

int CEC2014::Initialize(int func_id, int dim)
{
	func_id_ = func_id;
	dim_ = dim;
	cf_index_ = 0;
	cf_func_num_ = 1;
	ye = new real[dim_];
	ze = new real[dim_];
	x_bound = new real[dim_];
	pop_original_ = new real[dim_];

	switch (func_id_)
	{
	case 23:
		cf_func_num_ = 5;
		break;
	case 24:
		cf_func_num_ = 3;
		break;
	case 25:
		cf_func_num_ = 3;
		break;
	case 26:
		cf_func_num_ = 5;
		break;
	case 27:
		cf_func_num_ = 5;
		break;
	case 28:
		cf_func_num_ = 5;
		break;
	case 29:
		cf_func_num_ = 3;
		break;
	case 30:
		cf_func_num_ = 3;
		break;
	default:
		break;
	}


	FILE *fpt;
	char FileName[256];


	for (int i = 0; i<dim_; i++)
		x_bound[i] = 100.0;

	if (!(dim_ == 2 || dim_ == 10 || dim_ == 20 || dim_ == 30 || dim_ == 50 || dim_ == 100))
	{
		printf("\nError: Test functions are only defined for D=2,10,20,30,50,100.\n");
	}
	if (dim_ == 2 && ((func_id_ >= 17 && func_id_ <= 22) || (func_id_ >= 29 && func_id_ <= 30)))
	{
		printf("\nError: hf01,hf02,hf03,hf04,hf05,hf06,cf07&cf08 are NOT defined for D=2.\n");
	}

	/* Load Matrix Mdata*/
	sprintf(FileName, "input_data/M_%d_D%d.txt", func_id_, dim_);
	fpt = fopen(FileName, "r");
	if (fpt == NULL)
	{
		printf("\n Error: Cannot open input file for reading \n");
	}
	if (func_id_<23)
	{
		Mdata = new real[dim_ * dim_];
		if (Mdata == NULL)
			printf("\nError: there is insufficient memory available!\n");
		for (int i = 0; i<dim_*dim_; i++)
		{
#ifdef DOUBLE_PRECISION
			fscanf(fpt, "%lf", &Mdata[i]);
#else
			fscanf(fpt, "%f", &Mdata[i]);
#endif
		}
	}
	else
	{
		Mdata = new real[dim_ * dim_ * cf_func_num_];
		if (Mdata == NULL)
			printf("\nError: there is insufficient memory available!\n");
		for (int i = 0; i<cf_func_num_*dim_*dim_; i++)
		{
#ifdef DOUBLE_PRECISION
			fscanf(fpt, "%lf", &Mdata[i]);
#else
			fscanf(fpt, "%f", &Mdata[i]);
#endif

		}

	}
	fclose(fpt);

	/* Load shift_data */
	sprintf(FileName, "input_data/shift_data_%d.txt", func_id_);
	fpt = fopen(FileName, "r");
	if (fpt == NULL)
	{
		printf("\n Error: Cannot open input file for reading \n");
	}
	if (func_id_<23)
	{
		OShift = new real[dim_];
		if (OShift == NULL)
			printf("\nError: there is insufficient memory available!\n");
		for (int i = 0; i<dim_; i++)
		{
#ifdef DOUBLE_PRECISION
			fscanf(fpt, "%lf", &OShift[i]);
#else
			fscanf(fpt, "%f", &OShift[i]);
#endif
			
		}
	}
	else
	{
		OShift = new real[dim_ * cf_func_num_];
		if (OShift == NULL)
			printf("\nError: there is insufficient memory available!\n");
		for (int i = 0; i<cf_func_num_ - 1; i++)
		{
			for (int j = 0; j<dim_; j++)
			{
#ifdef DOUBLE_PRECISION
				fscanf(fpt, "%lf", &OShift[i*dim_ + j]);
#else
				fscanf(fpt, "%f", &OShift[i*dim_ + j]);
#endif
			}
			fscanf(fpt, "%*[^\n]%*c");
		}
		for (int j = 0; j<dim_; j++)
		{
#ifdef DOUBLE_PRECISION
			fscanf(fpt, "%lf", &OShift[(cf_func_num_ - 1)*dim_ + j]);
#else
			fscanf(fpt, "%f", &OShift[(cf_func_num_ - 1)*dim_ + j]);
#endif
		}

	}
	fclose(fpt);


	/* Load Shuffle_data */

	if (func_id_ >= 17 && func_id_ <= 22)
	{
		sprintf(FileName, "input_data/shuffle_data_%d_D%d.txt", func_id_, dim_);
		fpt = fopen(FileName, "r");
		if (fpt == NULL)
		{
			printf("\n Error: Cannot open input file for reading \n");
		}
		SS = new int[dim_ * cf_func_num_];
		if (SS == NULL)
			printf("\nError: there is insufficient memory available!\n");
		for (int i = 0; i<dim_; i++)
		{
			fscanf(fpt, "%d", &SS[i]);
		}
		fclose(fpt);
	}
	else if (func_id_ == 29 || func_id_ == 30)
	{
		sprintf(FileName, "input_data/shuffle_data_%d_D%d.txt", func_id_, dim_);
		fpt = fopen(FileName, "r");
		if (fpt == NULL)
		{
			printf("\n Error: Cannot open input file for reading \n");
		}
		SS = new int[dim_ * cf_func_num_];
		if (SS == NULL)
			printf("\nError: there is insufficient memory available!\n");
		for (int i = 0; i<dim_*cf_func_num_; i++)
		{
			fscanf(fpt, "%d", &SS[i]);
		}
		fclose(fpt);
	}
	return 0;
}

void CEC2014::sphere_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Sphere */
{
	int i;
	f[0] = 0.0;
	real *ze = new real[nx];
	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */

	for (i = 0; i<nx; i++)
	{
		ze[i] += fixedShift;
	}

	for (i = 0; i<nx; i++)
	{
		f[0] += ze[i] * ze[i];
	}
	delete []ze;
}


void CEC2014::ellips_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Ellipsoidal */
{
	int i;
	f[0] = 0.0;
	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */


	for (i = 0; i<nx; i++)
	{
		ze[i] += fixedShift;
	}
	for (i = 0; i<nx; i++)
	{
		real tmp = pow(10.0, 6.0*i / (nx - 1))*ze[i] * ze[i];
		f[0] += tmp;
	}
}

void CEC2014::bent_cigar_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Bent_Cigar */
{
	int i;
	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */

	for (i = 0; i<nx; i++)
	{
		ze[i] += fixedShift;
	}

	f[0] = ze[0] * ze[0];
	for (i = 1; i<nx; i++)
	{
		f[0] += pow(10.0, 6.0)*ze[i] * ze[i];
	}


}

void CEC2014::discus_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Discus */
{
	int i;
	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */
	f[0] = pow(10.0, 6.0)*ze[0] * ze[0];
	for (i = 0; i<nx; i++)
	{
		ze[i] += fixedShift;
	}
	for (i = 1; i<nx; i++)
	{
		f[0] += ze[i] * ze[i];
	}
}

void CEC2014::dif_powers_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Different Powers */
{
	int i;
	f[0] = 0.0;
	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */

	for (i = 0; i<nx; i++)
	{
		ze[i] += fixedShift;
	}

	for (i = 0; i<nx; i++)
	{
		f[0] += pow(fabs(ze[i]), 2 + 4 * i / (nx - 1));
	}
	f[0] = pow(f[0], 0.5);
}


void CEC2014::rosenbrock_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Rosenbrock's */
{
	int i;
	real tmp1, tmp2;
	f[0] = 0.0;
	real *ze = new real[nx];
	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */
	for (i = 0; i<nx; i++)
		ze[i] += fixedShift;//shift to orgin
	for (i = 0; i<nx - 1; i++)
	{
		//		ze[i + 1] += 1.0;//shift to orgin
		tmp1 = ze[i] * ze[i] - ze[i + 1];
		tmp2 = ze[i] - 1.0;
		f[0] += 100.0*tmp1*tmp1 + tmp2*tmp2;
	}
	delete[]ze;
}

void CEC2014::schaffer_F7_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Schwefel's 1.2  */
{
	int i;
	real tmp;
	f[0] = 0.0;
	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */
	for (i = 0; i<nx; i++)
	{
		ze[i] += fixedShift;
	}

	for (i = 0; i<nx - 1; i++)
	{
		ze[i] = pow(ye[i] * ye[i] + ye[i + 1] * ye[i + 1], 0.5);
		tmp = sin(50.0*pow(ze[i], 0.2));
		f[0] += pow(ze[i], 0.5) + pow(ze[i], 0.5)*tmp*tmp;
	}
	f[0] = f[0] * f[0] / (nx - 1) / (nx - 1);
}

void CEC2014::ackley_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Ackley's  */
{
	int i;
	real sum1, sum2;
	sum1 = 0.0;
	sum2 = 0.0;
	real *ze = new real[nx];
	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */

	for (i = 0; i<nx; i++)
	{
		ze[i] += fixedShift;
	}

	for (i = 0; i<nx; i++)
	{
		sum1 += ze[i] * ze[i];
		sum2 += cos(2.0*PI*ze[i]);
	}
	sum1 = -0.2*sqrt(sum1 / (real)nx);
	sum2 /= (real)nx;
	f[0] = M_E - 20.0*exp(sum1) + 20.0 - exp(sum2);
	delete[] ze;
}


// #include <omp.h>

void CEC2014::weierstrass_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Weierstrass's  */
{
	int i, j, k_max;
	real sum, sum2, a, b;
	a = 0.5;
	b = 3.0;
	k_max = 20;
	f[0] = 0.0;
	real *ze = new real[nx];

	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */

	// for (i = 0; i<nx; i++)
	// {
	// 		ze[i] += fixedShift;	
	// }
	real twopi = 2.0 * PI;
	// int thread_num = 4;
	// omp_set_num_threads(thread_num);
	// real sum_tmp_parallel[thread_num] = {0.0};

	// #pragma omp parallel
	// {
	// 	int id = omp_get_thread_num();	
	// 	#pragma omp for
	// 	for (i = 0; i<nx; i++)
	// 	{
	// 		ze[i] += fixedShift;
	// 		real temp = 0;
	// 		for (j = 0; j <= k_max; j++)
	// 		{
	// 			// sum += pow(a, j)*cos(2.0*PI*pow(b, j)*(ze[i] + 0.5));
	// 			temp += pow_a_values_weierstrass[j] * \
	// 				cos(twopi * pow_b_values_weierstrass[j] * (ze[i] + 0.5));
	// 			// sum2 += pow(a, j)*cos(2.0*PI*pow(b, j)*0.5);
	// 		}
	// 		sum_tmp_parallel[id] += temp;	
	// 	}	
	// }
	// for (int k = 0; k < thread_num; k++)
	// {
	// 	f[0] += sum_tmp_parallel[k];
	// }

	// printf("nx weierstrass %d\n", nx);
	for(i = 0; i < nx; i++)
	{
		ze[i] += fixedShift;
		sum = 0;
		for (j = 0; j <= k_max; j++)
		{
			sum += pow_a_values_weierstrass[j] * \
				cos(twopi * pow_b_values_weierstrass[j] * (ze[i] + 0.5));
		}
		f[0] += sum;
	}
	f[0] -= nx*sum2_weierstrass;
	delete[]ze;
}


void CEC2014::griewank_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Griewank's  */
{
	int i;
	real s, p;
	s = 0.0;
	p = 1.0;
	real *ze = new real[nx];
	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */

	for (i = 0; i<nx; i++)
	{
		ze[i] += fixedShift;
	}

	for (i = 0; i<nx; i++)
	{
		s += ze[i] * ze[i];
		p *= cos(ze[i] / sqrt(1.0 + i));
	}
	f[0] = 1.0 + s / 4000.0 - p;
	delete[]ze;
}

void CEC2014::rastrigin_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Rastrigin's  */
{
	int i;
	f[0] = 0.0;
	real *ze = new real[nx];

	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */

	for (i = 0; i<nx; i++)
	{
		ze[i] += fixedShift;
	}
	for (i = 0; i<nx; i++)
	{
		f[0] += (ze[i] * ze[i] - 10.0*cos(2.0*PI*ze[i]) + 10.0);
	}
	delete []ze;
}

void CEC2014::step_rastrigin_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Noncontinuous Rastrigin's  */
{
	int i;
	f[0] = 0.0;


	for (i = 0; i<nx; i++)
	{
		if (fabs(ye[i] - Os[i])>0.5)
			ye[i] = Os[i] + floor(2 * (ye[i] - Os[i]) + 0.5) / 2;
	}

	sr_func(x, ze, nx, Os, Mr, 5.12 / 100.0, s_flag, r_flag); /* shift and rotate */

	for (i = 0; i<nx; i++)
	{
		f[0] += (ze[i] * ze[i] - 10.0*cos(2.0*PI*ze[i]) + 10.0);
	}
}

void CEC2014::schwefel_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Schwefel's  */
{
	int i;
	real tmp;
	f[0] = 0.0;
	real *ze = new real[nx];
	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */

	for (i = 0; i<nx; i++)
	{
		ze[i] += fixedShift;
	}
	for (i = 0; i<nx; i++)
	{
		// ze[i] += 4.209687462275036e+002;
		if (ze[i]>500)
		{
			f[0] -= (500.0 - fmod(ze[i], 500))*sin(pow(500.0 - fmod(ze[i], 500), 0.5));
			tmp = (ze[i] - 500.0) / 100;
			f[0] += tmp*tmp / nx;
		}
		else if (ze[i]<-500)
		{
			f[0] -= (-500.0 + fmod(fabs(ze[i]), 500))*sin(pow(500.0 - fmod(fabs(ze[i]), 500), 0.5));
			tmp = (ze[i] + 500.0) / 100;
			f[0] += tmp*tmp / nx;
		}
		else
			f[0] -= ze[i] * sin(pow(fabs(ze[i]), 0.5));
	}
	f[0] += 4.189828872724338e+002*nx;
	delete[]ze;
}

void CEC2014::katsuura_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Katsuura  */
{
	int i, j;
	real temp, tmp1, tmp2, tmp3;
	f[0] = 1.0;
	tmp3 = pow(1.0*nx, 1.2);

	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */
	for (i = 0; i<nx; i++)
	{
		ze[i] += fixedShift;
	}
	for (i = 0; i<nx; i++)
	{
		temp = 0.0;
		for (j = 1; j <= 32; j++)
		{
			tmp1 = pow(2.0, j);
			tmp2 = tmp1*ze[i];
			temp += fabs(tmp2 - floor(tmp2 + 0.5)) / tmp1;
		}
		f[0] *= pow(1.0 + (i + 1)*temp, 10.0 / tmp3);
	}
	tmp1 = 10.0 / nx / nx;
	f[0] = f[0] * tmp1 - tmp1;

}

void CEC2014::bi_rastrigin_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Lunacek Bi_rastrigin Function */
{
	int i;
	real mu0 = 2.5, d = 1.0, s, mu1, tmp, tmp1, tmp2;
	real *tmpx;
	tmpx = (real *)malloc(sizeof(real)  *  nx);
	s = 1.0 - 1.0 / (2.0*pow(nx + 20.0, 0.5) - 8.2);
	mu1 = -pow((mu0*mu0 - d) / s, 0.5);

	if (s_flag == 1)
		shiftfunc(x, ye, nx, Os);
	else
	{
		for (i = 0; i<nx; i++)//shrink to the orginal search range
		{
			ye[i] = x[i];
		}
	}
	for (i = 0; i<nx; i++)//shrink to the orginal search range
	{
		ye[i] *= 10.0 / 100.0;
	}

	for (i = 0; i < nx; i++)
	{
		tmpx[i] = 2 * ye[i];
		if (Os[i] < 0.0)
			tmpx[i] *= -1.;
	}
	for (i = 0; i<nx; i++)
	{
		ze[i] = tmpx[i];
		tmpx[i] += mu0;
	}
	tmp1 = 0.0; tmp2 = 0.0;
	for (i = 0; i<nx; i++)
	{
		tmp = tmpx[i] - mu0;
		tmp1 += tmp*tmp;
		tmp = tmpx[i] - mu1;
		tmp2 += tmp*tmp;
	}
	tmp2 *= s;
	tmp2 += d*nx;
	tmp = 0.0;

	if (r_flag == 1)
	{
		rotatefunc(ze, ye, nx, Mr);
		for (i = 0; i<nx; i++)
		{
			tmp += cos(2.0*PI*ye[i]);
		}
		if (tmp1<tmp2)
			f[0] = tmp1;
		else
			f[0] = tmp2;
		f[0] += 10.0*(nx - tmp);
	}
	else
	{
		for (i = 0; i<nx; i++)
		{
			tmp += cos(2.0*PI*ze[i]);
		}
		if (tmp1<tmp2)
			f[0] = tmp1;
		else
			f[0] = tmp2;
		f[0] += 10.0*(nx - tmp);
	}

	free(tmpx);
}

void CEC2014::grie_rosen_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Griewank-Rosenbrock  */
{
	int i;
	real temp, tmp1, tmp2;
	f[0] = 0.0;

	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */
	for (i = 0; i<nx; i++)
	{
		ze[i] += fixedShift;
	}
	//for (i = 0; i<nx; i++)
	//	ze[i] += 1.0;//shift to orgin
	for (i = 0; i<nx - 1; i++)
	{
		//		ze[i + 1] += 1.0;//shift to orgin
		tmp1 = ze[i] * ze[i] - ze[i + 1];
		tmp2 = ze[i] - 1.0;
		temp = 100.0*tmp1*tmp1 + tmp2*tmp2;
		f[0] += (temp*temp) / 4000.0 - cos(temp) + 1.0;
	}
	tmp1 = ze[nx - 1] * ze[nx - 1] - ze[0];
	tmp2 = ze[nx - 1] - 1.0;
	temp = 100.0*tmp1*tmp1 + tmp2*tmp2;
	f[0] += (temp*temp) / 4000.0 - cos(temp) + 1.0;
}


void CEC2014::escaffer6_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* Expanded Scaffer??s F6  */
{
	int i;
	real temp1, temp2;

	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */

	f[0] = 0.0;
	for (i = 0; i<nx; i++)
	{
		ze[i] += fixedShift;
	}
	//for (i = 0; i<nx; i++)
	//	ze[i] += 1.0;//shift to orgin
	if (nx != 1)
	{
		for (i = 0; i < nx - 1; i++)
		{
			temp1 = sin(sqrt(ze[i] * ze[i] + ze[i + 1] * ze[i + 1]));
			temp1 = temp1*temp1;
			temp2 = 1.0 + 0.001*(ze[i] * ze[i] + ze[i + 1] * ze[i + 1]);
			f[0] += 0.5 + (temp1 - 0.5) / (temp2*temp2);
		}
		temp1 = sin(sqrt(ze[nx - 1] * ze[nx - 1] + ze[0] * ze[0]));
		temp1 = temp1*temp1;
		temp2 = 1.0 + 0.001*(ze[nx - 1] * ze[nx - 1] + ze[0] * ze[0]);
		f[0] += 0.5 + (temp1 - 0.5) / (temp2*temp2);
	}
	else
	{
		temp1 = sin(ze[0]);
		temp1 = temp1*temp1;
		temp2 = 1.0 + 0.001*(ze[0] * ze[0]);
		f[0] += 0.5 + (temp1 - 0.5) / (temp2*temp2);
	}
}

void CEC2014::happycat_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* HappyCat, provdided by Hans-Georg Beyer (HGB) */
/* original global optimum: [-1,-1,...,-1] */
{
	int i;
	real alpha, r2, sum_z;
	alpha = 1.0 / 8.0;

	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */
	for (i = 0; i<nx; i++)
	{
		ze[i] += fixedShift;
	}
	r2 = 0.0;
	sum_z = 0.0;
	for (i = 0; i<nx; i++)
	{
		ze[i] = ze[i] - 1.0;//shift to orgin

		r2 += ze[i] * ze[i];
		sum_z += ze[i];
	}
	f[0] = pow(fabs(r2 - nx), 2 * alpha) + (0.5*r2 + sum_z) / nx + 0.5;
}

void CEC2014::hgbat_func(real *x, real *f, int nx, real *Os, real *Mr, int s_flag, int r_flag, real fixedShift, real rate) /* HGBat, provdided by Hans-Georg Beyer (HGB)*/
/* original global optimum: [-1,-1,...,-1] */
{
	int i;
	real alpha, r2, sum_z;
	alpha = 1.0 / 4.0;

	sr_func(x, ze, nx, Os, Mr, rate, s_flag, r_flag); /* shift and rotate */
	for (i = 0; i<nx; i++)
	{
		ze[i] += fixedShift;
	}
	r2 = 0.0;
	sum_z = 0.0;
	for (i = 0; i<nx; i++)
	{
		ze[i] = ze[i] - 1.0;//shift to orgin
		r2 += ze[i] * ze[i];
		sum_z += ze[i];
	}
	f[0] = pow(fabs(pow(r2, 2.0) - pow(sum_z, 2.0)), 2 * alpha) + (0.5*r2 + sum_z) / nx + 0.5;
}

void CEC2014::hf01(real *x, real *f, int nx, real *Os, real *Mr, int *S, int s_flag, int r_flag) /* Hybrid Function 1 */
{
	int i, tmp, cf_num = 3;
	real fit[3];
	int G[3], G_nx[3];
	real Gp[3] = { 0.3, 0.3, 0.4 };

	tmp = 0;
	for (i = 0; i<cf_num - 1; i++)
	{
		G_nx[i] = ceil(Gp[i] * nx);
		tmp += G_nx[i];
	}
	G_nx[cf_num - 1] = nx - tmp;
	G[0] = 0;
	for (i = 1; i<cf_num; i++)
	{
		G[i] = G[i - 1] + G_nx[i - 1];
	}

	sr_func(x, ze, nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

	for (i = 0; i<nx; i++)
	{
		ye[i] = ze[S[i] - 1];
	}
	i = 0;
	schwefel_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 10.0);
	i = 1;
	rastrigin_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 5.12 / 100.0);
	i = 2;
	ellips_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 1);
	f[0] = 0.0;
	for (i = 0; i<cf_num; i++)
	{
		f[0] += fit[i];
	}
}

void CEC2014::hf02(real *x, real *f, int nx, real *Os, real *Mr, int *S, int s_flag, int r_flag) /* Hybrid Function 2 */
{
	int i, tmp, cf_num = 3;
	real fit[3];
	int G[3], G_nx[3];
	real Gp[3] = { 0.3, 0.3, 0.4 };

	tmp = 0;
	for (i = 0; i<cf_num - 1; i++)
	{
		G_nx[i] = ceil(Gp[i] * nx);
		tmp += G_nx[i];
	}
	G_nx[cf_num - 1] = nx - tmp;

	G[0] = 0;
	for (i = 1; i<cf_num; i++)
	{
		G[i] = G[i - 1] + G_nx[i - 1];
	}

	sr_func(x, ze, nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

	for (i = 0; i<nx; i++)
	{
		ye[i] = ze[S[i] - 1];
	}
	i = 0;
	bent_cigar_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 1);
	i = 1;
	hgbat_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 5.0 / 100.0);
	i = 2;
	rastrigin_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 5.12 / 100.0);

	f[0] = 0.0;
	for (i = 0; i<cf_num; i++)
	{
		f[0] += fit[i];
	}
}

void CEC2014::hf03(real *x, real *f, int nx, real *Os, real *Mr, int *S, int s_flag, int r_flag) /* Hybrid Function 3 */
{
	int i, tmp, cf_num = 4;
	real fit[4] = { 0 };
	int G[4], G_nx[4];
	real Gp[4] = { 0.2, 0.2, 0.3, 0.3 };

	tmp = 0;
	for (i = 0; i<cf_num - 1; i++)
	{
		G_nx[i] = ceil(Gp[i] * nx);
		tmp += G_nx[i];
	}
	G_nx[cf_num - 1] = nx - tmp;

	G[0] = 0;
	for (i = 1; i<cf_num; i++)
	{
		G[i] = G[i - 1] + G_nx[i - 1];
	}

	sr_func(x, ze, nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

	for (i = 0; i<nx; i++)
	{
		ye[i] = ze[S[i] - 1];
	}
	i = 0;
	griewank_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 600.0 / 100.0);
	i = 1;
	weierstrass_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 0.5 / 100.0);
	i = 2;
	rosenbrock_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 1, 2.048 / 100.0);
	i = 3;
	escaffer6_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 1);

	f[0] = 0.0;
	for (i = 0; i<cf_num; i++)
	{
		f[0] += fit[i];
	}
}

void CEC2014::hf04(real *x, real *f, int nx, real *Os, real *Mr, int *S, int s_flag, int r_flag) /* Hybrid Function 4 */
{
	int i, tmp, cf_num = 4;
	real fit[4] = { 0 };
	int G[4], G_nx[4];
	real Gp[4] = { 0.2, 0.2, 0.3, 0.3 };

	tmp = 0;
	for (i = 0; i<cf_num - 1; i++)
	{
		G_nx[i] = ceil(Gp[i] * nx);
		tmp += G_nx[i];
	}
	G_nx[cf_num - 1] = nx - tmp;

	G[0] = 0;
	for (i = 1; i<cf_num; i++)
	{
		G[i] = G[i - 1] + G_nx[i - 1];
	}

	sr_func(x, ze, nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

	for (i = 0; i<nx; i++)
	{
		ye[i] = ze[S[i] - 1];
	}
	i = 0;
	hgbat_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 5.0 / 100.0);
	i = 1;
	discus_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 1);
	i = 2;
	grie_rosen_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 1, 5.0 / 100.0);
	i = 3;
	rastrigin_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 5.12 / 100.0);

	f[0] = 0.0;
	for (i = 0; i<cf_num; i++)
	{
		f[0] += fit[i];
	}
}
void CEC2014::hf05(real *x, real *f, int nx, real *Os, real *Mr, int *S, int s_flag, int r_flag) /* Hybrid Function 5 */
{
	int i, tmp, cf_num = 5;
	real fit[5] = { 0 };
	int G[5], G_nx[5];
	real Gp[5] = { 0.1, 0.2, 0.2, 0.2, 0.3 };

	tmp = 0;
	for (i = 0; i<cf_num - 1; i++)
	{
		G_nx[i] = ceil(Gp[i] * nx);
		tmp += G_nx[i];
	}
	G_nx[cf_num - 1] = nx - tmp;

	G[0] = 0;
	for (i = 1; i<cf_num; i++)
	{
		G[i] = G[i - 1] + G_nx[i - 1];
	}

	sr_func(x, ze, nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

	for (i = 0; i<nx; i++)
	{
		ye[i] = ze[S[i] - 1];
	}
	i = 0;
	escaffer6_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 1);
	i = 1;
	hgbat_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 5.0 / 100.0);
	i = 2;
	rosenbrock_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 1, 2.048 / 100.0);
	i = 3;
	schwefel_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 1000.0 / 100.0);
	i = 4;
	ellips_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 1);

	f[0] = 0.0;
	for (i = 0; i<cf_num; i++)
	{
		f[0] += fit[i];
	}
}

void CEC2014::hf06(real *x, real *f, int nx, real *Os, real *Mr, int *S, int s_flag, int r_flag) /* Hybrid Function 6 */
{
	int i, tmp, cf_num = 5;
	real fit[5] = { 0 };
	int G[5], G_nx[5];
	real Gp[5] = { 0.1, 0.2, 0.2, 0.2, 0.2 };

	tmp = 0;
	for (i = 0; i<cf_num - 1; i++)
	{
		G_nx[i] = ceil(Gp[i] * nx);
		tmp += G_nx[i];
	}
	G_nx[cf_num - 1] = nx - tmp;

	G[0] = 0;
	for (i = 1; i<cf_num; i++)
	{
		G[i] = G[i - 1] + G_nx[i - 1];
	}

	sr_func(x, ze, nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

	for (i = 0; i<nx; i++)
	{
		ye[i] = ze[S[i] - 1];
	}
	i = 0;
	katsuura_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 5.0 / 100.0);
	i = 1;
	happycat_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 5.0 / 100.0);
	i = 2;
	grie_rosen_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 1, 5.0 / 100.0);
	i = 3;
	schwefel_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 1000.0 / 100.0);
	i = 4;
	ackley_func(&ye[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0, 0, 1);

	f[0] = 0.0;
	for (i = 0; i<cf_num; i++)
	{
		f[0] += fit[i];
	}
}

void CEC2014::cf01(real *x, real *f, int nx, real *Os, real *Mr, int r_flag) /* Composition Function 1 */
{
	int i, cf_num = 5;
	real fit[5];
	real delta[5] = { 10, 20, 30, 40, 50 };
	real bias[5] = { 0, 100, 200, 300, 400 };

	i = 0;
	rosenbrock_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, 1, 1, 2.048 / 100.0);
	fit[i] = 10000 * fit[i] / 1e+4;
	i = 1;
	ellips_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, 1, 0, 1);
	fit[i] = 10000 * fit[i] / 1e+10;
	i = 2;
	bent_cigar_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, 1, 0, 1);
	fit[i] = 10000 * fit[i] / 1e+30;
	i = 3;
	discus_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, 1, 0, 1);
	fit[i] = 10000 * fit[i] / 1e+10;
	i = 4;
	ellips_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, 0, 0, 1);
	fit[i] = 10000 * fit[i] / 1e+10;
	*f = fit[0] + fit[1] + fit[2] + fit[3] + fit[4];
	cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}

void CEC2014::cf02(real *x, real *f, int nx, real *Os, real *Mr, int r_flag) /* Composition Function 2 */
{
	int i, cf_num = 3;
	real fit[3];
	real delta[3] = { 20, 20, 20 };
	real bias[3] = { 0, 100, 200 };

	i = 0;
	schwefel_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, 0, 0, 1000.0 / 100.0);
	i = 1;
	rastrigin_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 5.12 / 100.0);
	i = 2;
	hgbat_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 5.0 / 100.0);
	cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}

void CEC2014::cf03(real *x, real *f, int nx, real *Os, real *Mr, int r_flag) /* Composition Function 3 */
{
	int i, cf_num = 3;
	real fit[3];
	real delta[3] = { 10, 30, 50 };
	real bias[3] = { 0, 100, 200 };
	i = 0;
	schwefel_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 1000.0 / 100.0);
	fit[i] = 1000 * fit[i] / 4e+3;
	i = 1;
	rastrigin_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 5.12 / 100.0);
	fit[i] = 1000 * fit[i] / 1e+3;
	i = 2;
	ellips_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 1);
	fit[i] = 1000 * fit[i] / 1e+10;
	cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}

void CEC2014::cf04(real *x, real *f, int nx, real *Os, real *Mr, int r_flag) /* Composition Function 4 */
{
	int i, cf_num = 5;
	real fit[5];
	real delta[5] = { 10, 10, 10, 10, 10 };
	real bias[5] = { 0, 100, 200, 300, 400 };
	i = 0;
	schwefel_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 1000.0 / 100.0);
	fit[i] = 1000 * fit[i] / 4e+3;
	i = 1;
	happycat_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 5.0 / 100.0);
	fit[i] = 1000 * fit[i] / 1e+3;
	i = 2;
	ellips_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 1);
	fit[i] = 1000 * fit[i] / 1e+10;
	i = 3;
	weierstrass_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 0.5 / 100.0);
	fit[i] = 1000 * fit[i] / 400;
	i = 4;
	griewank_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 600.0 / 100.0);
	fit[i] = 1000 * fit[i] / 100;
	cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}

void CEC2014::cf05(real *x, real *f, int nx, real *Os, real *Mr, int r_flag) /* Composition Function 4 */
{
	int i, cf_num = 5;
	real fit[5];
	real delta[5] = { 10, 10, 10, 20, 20 };
	real bias[5] = { 0, 100, 200, 300, 400 };
	i = 0;
	hgbat_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 5.0 / 100.0);
	fit[i] = 10000 * fit[i] / 1000;
	i = 1;
	rastrigin_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 5.12 / 100.0);
	fit[i] = 10000 * fit[i] / 1e+3;
	i = 2;
	schwefel_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 1000.0 / 100.0);
	fit[i] = 10000 * fit[i] / 4e+3;
	i = 3;
	weierstrass_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 0.5 / 100.0);
	fit[i] = 10000 * fit[i] / 400;
	i = 4;
	ellips_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 1);
	fit[i] = 10000 * fit[i] / 1e+10;
	cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}

void CEC2014::cf06(real *x, real *f, int nx, real *Os, real *Mr, int r_flag) /* Composition Function 6 */
{
	int i, cf_num = 5;
	real fit[5];
	real delta[5] = { 10, 20, 30, 40, 50 };
	real bias[5] = { 0, 100, 200, 300, 400 };
	i = 0;
	grie_rosen_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 1, 5.0 / 100.0);
	fit[i] = 10000 * fit[i] / 4e+3;
	i = 1;
	happycat_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 5.0 / 100.0);
	fit[i] = 10000 * fit[i] / 1e+3;
	i = 2;
	schwefel_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 1000.0 / 100.0);
	fit[i] = 10000 * fit[i] / 4e+3;
	i = 3;
	escaffer6_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 1);
	fit[i] = 10000 * fit[i] / 2e+7;
	i = 4;
	ellips_func(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], 1, r_flag, 0, 1);
	fit[i] = 10000 * fit[i] / 1e+10;
	cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}

void CEC2014::cf07(real *x, real *f, int nx, real *Os, real *Mr, int *SS, int r_flag) /* Composition Function 7 */
{
	int i, cf_num = 3;
	real fit[3];
	real delta[3] = { 10, 30, 50 };
	real bias[3] = { 0, 100, 200 };
	i = 0;
	hf01(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], &SS[i*nx], 1, r_flag);
	i = 1;
	hf02(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], &SS[i*nx], 1, r_flag);
	i = 2;
	hf03(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], &SS[i*nx], 1, r_flag);
	cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}

void CEC2014::cf08(real *x, real *f, int nx, real *Os, real *Mr, int *SS, int r_flag) /* Composition Function 8 */
{
	int i, cf_num = 3;
	real fit[3];
	real delta[3] = { 10, 30, 50 };
	real bias[3] = { 0, 100, 200 };
	i = 0;
	hf04(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], &SS[i*nx], 1, r_flag);
	i = 1;
	hf05(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], &SS[i*nx], 1, r_flag);
	i = 2;
	hf06(x, &fit[i], nx, &Os[i*nx], &Mr[i*nx*nx], &SS[i*nx], 1, r_flag);
	cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}



void CEC2014::shiftfunc(real *x, real *xshift, int nx, real *Os)
{
	int i;
	for (i = 0; i<nx; i++)
	{
		xshift[i] = x[i] - Os[i];
	}
}

void CEC2014::rotatefunc(real *x, real *xrot, int nx, real *Mr)
{
	int i, j;
	for (i = 0; i<nx; i++)
	{
		xrot[i] = 0;
		for (j = 0; j<nx; j++)
		{
			xrot[i] = xrot[i] + x[j] * Mr[i*nx + j];
		}
	}
}
void CEC2014::sr_func(real *x, real *sr_x, int nx, real *Os, real *Mr, real sh_rate, int s_flag, int r_flag) /* shift and rotate */
{
	int i;
	real *ye = new real[nx];
	if (s_flag == 1)
	{
		if (r_flag == 1)
		{
			shiftfunc(x, ye, nx, Os);
			for (i = 0; i<nx; i++)//shrink to the original search range
			{
				ye[i] = ye[i] * sh_rate;
			}
			rotatefunc(ye, sr_x, nx, Mr);
		}
		else
		{
			shiftfunc(x, sr_x, nx, Os);
			for (i = 0; i<nx; i++)//shrink to the original search range
			{
				sr_x[i] = sr_x[i] * sh_rate;
			}
		}
	}
	else
	{

		if (r_flag == 1)
		{
			for (i = 0; i<nx; i++)//shrink to the original search range
			{
				ye[i] = x[i] * sh_rate;
			}
			rotatefunc(ye, sr_x, nx, Mr);
		}
		else
			for (i = 0; i<nx; i++)//shrink to the original search range
			{
				sr_x[i] = x[i] * sh_rate;
			}
	}
	delete[]ye;
}

void CEC2014::asyfunc(real *x, real *xasy, int nx, real beta)
{
	int i;
	for (i = 0; i<nx; i++)
	{
		if (x[i]>0)
			xasy[i] = pow(x[i], 1.0 + beta*i / (nx - 1)*pow(x[i], 0.5));
	}
}

void CEC2014::oszfunc(real *x, real *xosz, int nx)
{
	int i, sx;
	real c1, c2, xx;
	for (i = 0; i<nx; i++)
	{
		if (i == 0 || i == nx - 1)
		{
			if (x[i] != 0)
				xx = log(fabs(x[i]));
			if (x[i]>0)
			{
				c1 = 10;
				c2 = 7.9;
			}
			else
			{
				c1 = 5.5;
				c2 = 3.1;
			}
			if (x[i]>0)
				sx = 1;
			else if (x[i] == 0)
				sx = 0;
			else
				sx = -1;
			xosz[i] = sx*exp(xx + 0.049*(sin(c1*xx) + sin(c2*xx)));
		}
		else
			xosz[i] = x[i];
	}
}


void CEC2014::cf_cal(real *x, real *f, int nx, real *Os, real * delta, real * bias, real * fit, int cf_num)
{
	int i, j;
	real *w;
	real w_max = 0, w_sum = 0;
	w = (real *)malloc(cf_num * sizeof(real));
	for (i = 0; i<cf_num; i++)
	{
		fit[i] += bias[i];
		w[i] = 0;
		for (j = 0; j<nx; j++)
		{
			w[i] += pow(x[j] - Os[i*nx + j], 2.0);
		}
		if (w[i] != 0)
			w[i] = pow(1.0 / w[i], 0.5)*exp(-w[i] / 2.0 / nx / pow(delta[i], 2.0));
		else
			w[i] = INF;
		if (w[i]>w_max)
			w_max = w[i];
	}

	for (i = 0; i<cf_num; i++)
	{
		w_sum = w_sum + w[i];
	}
	if (w_max == 0)
	{
		for (i = 0; i<cf_num; i++)
			w[i] = 1;
		w_sum = cf_num;
	}
	f[0] = 0.0;
	for (i = 0; i<cf_num; i++)
	{
		f[0] = f[0] + w[i] / w_sum *fit[i];
	}
	free(w);
}


