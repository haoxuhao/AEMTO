#include "random.h"
#include <functional>
#include <algorithm>

Random::Random():distribution(0, RAND_MAX)
{

}


Random::~Random()
{

}

/**
 * Return a 2D random matrix, range from [0, 1]
 */
vector<vector<real>> Random::RandReal2D(int r, int c)
{
	vector<vector<real>> ret;
	for (int i = 0; i < r; i++)
	{
		vector<real> x;
		for (int j = 0; j < c; j++)
		{
			x.push_back(RandRealUnif(0, 1));
		}
		ret.push_back(x);
	}
	return ret;
}

int Random::RandIntUnif(int min_value, int max_value)
{
	//int rand_num = distribution(generator);
	if (min_value != max_value)
		return min_value + rand() % (max_value - min_value + 1);
	else
		return min_value;
}
real Random::RandRealUnif(real min_value, real max_value)
{
	//int rand_num = distribution(generator);
	// auto dice = std::bind(distribution, generator);
	if (min_value != max_value)
		return min_value + rand() / ((real) RAND_MAX + 0.0) * (max_value - min_value + 0.0);
	else
		return min_value;
}
real Random::RandRealNormal(real u, real std)
{
	std::normal_distribution<double> distribution(u, std);
	return distribution(generator);
}

vector<int> Random::Permutate(int arrary_length, int permutation_length)
{
	assert((arrary_length > 0 || permutation_length >= 0) && "Assert error: array_length <= 0 or permutation_length < 0");
	
	if(permutation_length > arrary_length)
	{
		permutation_length = arrary_length;
	}

	vector<int> permutate_index;

	// if(permutation_length == 0) return permutate_index;

	vector<int> global_perm(arrary_length);
	for (int local_ind_individual = 0; local_ind_individual < arrary_length; local_ind_individual++)
		global_perm[local_ind_individual] = local_ind_individual;

	int tmp = 0;
	int i = arrary_length;

	while (i > arrary_length - permutation_length)     //pm_depth is the number of random indices wanted (must be <= NP)
	{
		tmp = RandIntUnif(0, (i - 1));
		permutate_index.push_back(global_perm[tmp]);
		global_perm[tmp] = global_perm[i - 1];
		i--;
	}

	return permutate_index;
}

vector<int>	Random::Permutate(int arrary_length, int permutation_length, vector<int> &avoid_index)
{
	assert((arrary_length > 0 || permutation_length >= 0) && "Assert error: array_length <= 0 or permutation_length < 0");
	if(permutation_length > arrary_length)
	{
		printf("permutation_length=%d\arrary_length=%d\t", permutation_length, arrary_length);
		permutation_length = arrary_length;
	}
	vector<int> permutate_index;
	vector<int> global_perm(arrary_length);
	for (int local_ind_individual = 0; local_ind_individual < arrary_length; local_ind_individual++)
		global_perm[local_ind_individual] = local_ind_individual;

	int tmp = 0;
	int i = arrary_length;

	while (i > arrary_length - permutation_length)     //pm_depth is the number of random indices wanted (must be <= NP)
	{
		tmp = RandIntUnif(0, (i - 1));
        for (int j = 0; j < avoid_index.size(); j++)
        {
            if (global_perm[tmp] == avoid_index[i])
            {
                j = 0;
                tmp = RandIntUnif(0, (i - 1));
            }

        }

		permutate_index.push_back(global_perm[tmp]);
		global_perm[tmp] = global_perm[i - 1];
		i--;
	}

	return permutate_index;
}

int Random::Permutate(vector<int> & permutate_index, int arrary_length, int permutation_length)
{
	assert((arrary_length > 0 || permutation_length >= 0) && "Assert error: array_length <= 0 or permutation_length < 0");
	if(permutation_length > arrary_length)
	{
		printf("permutation_length=%d\arrary_length=%d\t", permutation_length, arrary_length);
		permutation_length = arrary_length;
	}
	vector<int> global_perm(arrary_length);
	for (int local_ind_individual = 0; local_ind_individual < arrary_length; local_ind_individual++)
		global_perm[local_ind_individual] = local_ind_individual;

	int tmp = 0;
	int i = arrary_length;

	while (i > arrary_length - permutation_length)     //pm_depth is the number of random indices wanted (must be <= NP)
	{
		tmp = RandIntUnif(0, (i - 1));
		permutate_index.push_back(global_perm[tmp]);
		global_perm[tmp] = global_perm[i - 1];
		i--;
	}
	return 0;

}

int Random::Permutate(int * permutate_index, int arrary_length, int permutation_length)
{
	assert((arrary_length > 0 || permutation_length >= 0) && "Assert error: array_length <= 0 or permutation_length < 0");
	if(permutation_length > arrary_length)
	{
		printf("permutation_length=%d\arrary_length=%d\t", permutation_length, arrary_length);
		permutation_length = arrary_length;
	}
	vector<int> global_perm(arrary_length);
	for (int local_ind_individual = 0; local_ind_individual < arrary_length; local_ind_individual++)
		global_perm[local_ind_individual] = local_ind_individual;

	int tmp = 0;
	int i = arrary_length;
	int count = 0;
	while (i > arrary_length - permutation_length)     //pm_depth is the number of random indices wanted (must be <= NP)
	{
		tmp = RandIntUnif(0, (i - 1));
		permutate_index[count] = global_perm[tmp];
		global_perm[tmp] = global_perm[i - 1];
		i--;
		count++;
	}
	return 0;
}
/**
 * return selected index
 */
int Random::roulette_sampling(vector<real> &pdf)
{
	real p = RandRealUnif(0, 1);
	real sum = 0;
	
	for (int i = 0; i < pdf.size(); i++)
	{
		if (i == (pdf.size() - 1))
		{
			return i;
		}
		if (p >= sum && p < (sum + pdf[i]))
		{
			return i;
		}
		sum += pdf[i];
	}
	return -1;
}
vector<int> Random::roulette_sample(vector<real> &fitnesses, int num)
{
    assert(num >= 0 && "num >= 0.");
    vector<int> ret_indices;
	int N = fitnesses.size();
    if (N == 0 || num == 0)
    {
        return ret_indices;
    }
    auto min_loc = min_element(fitnesses.begin(), fitnesses.end(),
        [](real l, real r){ return l < r; }
    );
    real min_value = *min_loc;
    vector<real> relative_len(N, 1.0);
    if(min_value != 0)
	{
        for(int i = 0; i < N; i++)
        {
            relative_len[i] = min_value / (fitnesses.at(i) + 1e-12); // minimization only
        }
    }
    real sum = 0;
    vector<real> prob_arr(N, 0.0);
    for (int i = 0; i < N; i++)
    {
        sum += relative_len[i];
        prob_arr[i] = sum;
    }
    while (ret_indices.size() != num)
    {
        real rand_value = RandRealUnif(0, sum);
        auto loc = lower_bound(prob_arr.begin(), prob_arr.end(), rand_value);
        int loc_index = distance(prob_arr.begin(), loc);
        ret_indices.push_back(loc_index);
    }
    return ret_indices;
}