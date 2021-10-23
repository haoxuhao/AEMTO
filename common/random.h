#ifndef __RANDOM_H__
#define __RANDOM_H__
#include "config.h"
#include <random>

class Random
{
public:
	Random(){};
	~Random(){};
  	vector<vector<Real>> RandReal2D(int r, int c);
	vector<int>	 Permutate(int arrary_length, int permutation_length);
	vector<int>	 Permutate(int arrary_length, int permutation_length, vector<int> &avoid_index);

	int RandIntUnif(int min_value, int max_value);
	Real RandRealUnif(Real min_value, Real max_value);
	Real RandRealNormal(Real u, Real std);

	int roulette_sampling(vector<Real> &pdf);
	vector<int> roulette_sample(vector<Real> &fitnesses, int num);
	vector<int> sus_sample(const vector<Real> &pdf, int num);
	unordered_map<int, int> sus_sampleV2(const vector<Real> &pdf, int num);
};

#endif
