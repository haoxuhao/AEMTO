#ifndef __RANDOM_H__
#define __RANDOM_H__
#include "config.h"
#include <random>

class Random
{
public:
	Random(){};
	~Random(){};
  	vector<vector<real>> RandReal2D(int r, int c);
	vector<int>	 Permutate(int arrary_length, int permutation_length);
	vector<int>	 Permutate(int arrary_length, int permutation_length, vector<int> &avoid_index);

	int RandIntUnif(int min_value, int max_value);
	real RandRealUnif(real min_value, real max_value);
	real RandRealNormal(real u, real std);

	int roulette_sampling(vector<real> &pdf);
	vector<int> roulette_sample(vector<real> &fitnesses, int num);
	vector<int> sus_sample(const vector<real> &pdf, int num);
	unordered_map<int, int> sus_sampleV2(const vector<real> &pdf, int num);
};

#endif
