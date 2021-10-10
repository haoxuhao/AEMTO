#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
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


#ifndef RECORED_DETAILS
#define RECORED_DETAILS
#endif
// #define DEEPINSIGHT


#ifndef DUAL_CONTROL
#define DUAL_CONTROL
#endif


#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
	typedef double real;
	#define REAL_MAX DBL_MAX
	#define REAL_MIN DBL_MIN
#else
	typedef float real;
	#define REAL_MAX FLT_MAX
	#define REAL_MIN FLT_MIN
#endif

typedef unsigned int uint;

struct Individual
{
	vector<real> elements;
	real fitness_value;
	int skill_factor;
};

typedef vector<Individual> Population;

//the knowledge transfer type
enum KTType {
	INSERT,
	UNI_CROSSOVER,
	DE_BASE_VEC,
};

struct ProblemInfo
{
	int dim;
	int task_id;
	int func_id;
	int run_ID;
	int total_runs;
	int max_base_FEs;
	int seed;
	int computing_time;
	string problem_def;

	real max_bound; //search max bound
	real min_bound; //search min bound
	real o_min_bound;  //origin min bound
	real o_max_bound;  //origin max bound
	real *d_transferd_elements_addr; //store the address for the transfered elements

	//shift and rotation data
	string shift_data_root;
	string shift_data_prefix;
	string rotation_data_root;
	string rotation_data_prefix;
	string bin_root;
		
	int is_rotate;
	int calc_dim; 
	int current_pop_size;
};

struct NodeInfo
{
    int task_ID;
	int task_type;
	int node_ID;//rank id
	int node_num;//rank num
	int nodes;//compute nodes
	int GPU_num;
	int GPU_ID;
	vector<int> tasks_curr;
	unordered_map<int, int> taskid_rank_map;
};

struct AdaParam
{
	real alpha;
	real beta;
	real Delta_T;
	real pbase;
	int sample_batch;
};
struct ParallelParam
{
	bool sync_gen;
	bool wait_pop;
	int sync_gens;
	bool use_buffer;
	int cuda_EA_delta; //delta generations for cuda EA each time
};
struct RunParam
{
	int total_runs;
	long FEs;
};

struct IslandInfo
{
	int island_size;
	int island_num;
	int comm_rank;
	int exec_rank;
	int island_ID;
	vector<int> task_ids;
	vector<int> global_islands_comm_ranks;
	unordered_map<int, int> comm_core_island_id_table;
	int exec_rank_0;
	string log_dir;
	string results_dir;
	string results_subdir;
	
	// MTO info
	real import_rate;
	real export_rate;
	int interval;
	int export_interval;
	int import_interval;
	real export_prob;
	real import_prob;

    real buffer_capacity;
	real connection_rate;
    string regroup_option;
    string migration_topology;
	string import_strategy;
	int select_num;
	int reward_self;
	string ada_import_strategy;
	real 	ada_import_epsilon;
	int ada_import_prob;
	real upper_import_prob;
	real lower_import_prob;
	string export_strategy;
	string emmigration_strategy;
	string buffer_manage;
	KTType transfer_type;

	AdaParam ada_param;
	ParallelParam pmto_param;
	RunParam run_param;
	string buffer_sampling;
	int print_interval;
	bool	record_details;
};

#endif

