#ifndef __RANDOM_H__
#define __RANDOM_H__
// #pragma once
#include "config.h"
#include <random>

class Random
{
public:
	Random();
	~Random();
	std::uniform_int_distribution<int> distribution;
	std::default_random_engine generator;
  	
	int			           Permutate(vector<int> & requested_island_ID, int arrary_length, int permutation_length);
	vector<int>	           Permutate(int arrary_length, int permutation_length);
	vector<int>	           Permutate(int arrary_length, int permutation_length, vector<int> &avoid_index);

	int 			Permutate(int * permutate_index, int arrary_length, int permutation_length);
	int			     RandIntUnif(int min_value, int max_value);
	real 			RandRealUnif(real min_value, real max_value);
	vector<vector<real> > 	RandReal2D(int r, int c);
	real 			RandRealNormal(real u, real std);

	int roulette_sampling(vector<real> &pdf);
	vector<int> roulette_sample(vector<real> &fitnesses, int num);
};

#endif
