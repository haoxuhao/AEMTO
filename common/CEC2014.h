#ifndef __CEC2014_H__
#define __CEC2014_H__

#include <math.h>
#include "config.h"
using namespace std;
#define INF 1.0e99
#define EPS 1.0e-14
#ifndef M_E
#define M_E  2.7182818284590452353602874713526625
#endif
#ifndef PI
#define PI 3.1415926535897932384626433832795029
#endif

class  CEC2014
{
protected:
	void sphere_func(real *, real *, int, real *, real *, int, int, real, real); /* Sphere */
	void ellips_func(real *, real *, int, real *, real *, int, int, real, real); /* Ellipsoidal */
	void bent_cigar_func(real *, real *, int, real *, real *, int, int, real, real); /* Discus */
	void discus_func(real *, real *, int, real *, real *, int, int, real, real);  /* Bent_Cigar */
	void dif_powers_func(real *, real *, int, real *, real *, int, int, real, real);  /* Different Powers */
	void rosenbrock_func(real *, real *, int, real *, real *, int, int, real, real); /* Rosenbrock's */
	void schaffer_F7_func(real *, real *, int, real *, real *, int, int, real, real); /* Schwefel's F7 */
	void ackley_func(real *, real *, int, real *, real *, int, int, real, real); /* Ackley's */
	void rastrigin_func(real *, real *, int, real *, real *, int, int, real, real); /* Rastrigin's  */
	void weierstrass_func(real *, real *, int, real *, real *, int, int, real, real); /* Weierstrass's  */
	void griewank_func(real *, real *, int, real *, real *, int, int, real, real); /* Griewank's  */
	void schwefel_func(real *, real *, int, real *, real *, int, int, real, real); /* Schwefel's */
	void katsuura_func(real *, real *, int, real *, real *, int, int, real, real); /* Katsuura */
	void bi_rastrigin_func(real *, real *, int, real *, real *, int, int, real, real); /* Lunacek Bi_rastrigin */
	void grie_rosen_func(real *, real *, int, real *, real *, int, int, real, real); /* Griewank-Rosenbrock  */
	void escaffer6_func(real *, real *, int, real *, real *, int, int, real, real); /* Expanded Scaffer??s F6  */
	void step_rastrigin_func(real *, real *, int, real *, real *, int, int, real, real); /* Noncontinuous Rastrigin's  */
	void happycat_func(real *, real *, int, real *, real *, int, int, real, real); /* HappyCat */
	void hgbat_func(real *, real *, int, real *, real *, int, int, real, real); /* HGBat  */

	void hf01(real *, real *, int, real *, real *, int *, int, int); /* Hybrid Function 1 */
	void hf02(real *, real *, int, real *, real *, int *, int, int); /* Hybrid Function 2 */
	void hf03(real *, real *, int, real *, real *, int *, int, int); /* Hybrid Function 3 */
	void hf04(real *, real *, int, real *, real *, int *, int, int); /* Hybrid Function 4 */
	void hf05(real *, real *, int, real *, real *, int *, int, int); /* Hybrid Function 5 */
	void hf06(real *, real *, int, real *, real *, int *, int, int); /* Hybrid Function 6 */

	void cf01(real *, real *, int, real *, real *, int); /* Composition Function 1 */
	void cf02(real *, real *, int, real *, real *, int); /* Composition Function 2 */
	void cf03(real *, real *, int, real *, real *, int); /* Composition Function 3 */
	void cf04(real *, real *, int, real *, real *, int); /* Composition Function 4 */
	void cf05(real *, real *, int, real *, real *, int); /* Composition Function 5 */
	void cf06(real *, real *, int, real *, real *, int); /* Composition Function 6 */
	void cf07(real *, real *, int, real *, real *, int *, int); /* Composition Function 7 */
	void cf08(real *, real *, int, real *, real *, int *, int); /* Composition Function 8 */

	void shiftfunc(real*, real*, int, real*);
	void rotatefunc(real*, real*, int, real*);
	void sr_func(real *, real *, int, real*, real*, real, int, int); /* shift and rotate */
	void asyfunc(real *, real *x, int, real);
	void oszfunc(real *, real *, int);
	void cf_cal(real *, real *, int, real *, real *, real *, real *, int);

	real 							*OShift;
	real							*Mdata;
	real							*ye;
	real							*ze;
	real							*x_bound;
	int 								*SS;

	int									func_id_;
	int            			dim_;
	int 								cf_func_num_;
	int                 cf_index_;

	real 							*pop_original_;
	/**
	 * Pre compute the constant for
	 * accelerating the Weierstrass eavluation
	 */
	real sum2_weierstrass;
	vector<real> pow_a_values_weierstrass;
	vector<real> pow_b_values_weierstrass;

public:
	CEC2014();			//construction function of CEC2014, it sets member variables
	~CEC2014();
	int								Initialize(int func_id, int dim);
	int								Uninitialize();
	virtual real					EvaluateFitness(const vector<real> & elements);
	real 							EvaluateFitness(real * elements);
};

#endif
