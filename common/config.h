#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <time.h>
#include <numeric>
#include <string>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <float.h>
#include <limits.h>
#include <assert.h>
#include <list>
#include <stdexcept>

using namespace std;

#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
	typedef double Real;
	#define REAL_MAX DBL_MAX
	#define REAL_MIN DBL_MIN
#else
	typedef float Real;
	#define REAL_MAX FLT_MAX
	#define REAL_MIN FLT_MIN
#endif

typedef unsigned int uint;

struct Individual
{
	vector<Real> elements;
	Real fitness_value;
	int skill_factor;
};

typedef vector<Individual> Population;

struct Args
{
    int total_runs {1};
    int record_interval {1};
	int G_max {100};
	int UDim {50};
	int popsize {100};
    vector<int> total_tasks;
	string problem_set;
	string problem_name;
    string params_file;
    string results_dir;
    string results_subdir;
    bool MTO {true};
};

struct ProblemInfo
{
	int dim;
	int task_id;
	int run_ID;
	int total_runs;
	string problem_def;
	string benchfunc_name;

	Real max_bound; //search max bound
	Real min_bound; //search min bound

	//shift and rotation data
	string shift_data_file;
	string rotation_data_file;
	string arm_data_file;
		
	int is_rotate;
	int calc_dim; 
};

struct IslandInfo
{
	int island_size;
};

#endif

