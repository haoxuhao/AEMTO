//Many-task Evolutionary Framework (implemented using DE, MaTDE) 
//Related paper: Y. Chen, J. Zhong, L. Feng and J. Zhang. "An Adaptive Archive-Based Evolutionary Framework for Many-Task Optimization."
//Coded by: Yongliang Chen
//South China University of Technology, Guangzhou, China
//Last Update: 2019/06/05
#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "time.h"
#include "string.h"
#include <iostream>
#include <vector>
#include <ctime>
#include <sstream>

using namespace std;


#define	MAX_NVARS	50			//The maximum dimension of the tasks.
#define	POPSIZE		100			//Population size for each sub-population
#define MAX_OBJS	1			//Maximum number of objectives
#define MAX_GEN		1000		//Maximum generation
#define RECORD_FRE	50	//Interval of recording
#define RUNS	20             //Total runs for each Problem
#define K_ARCHIVE_SIZE	(POPSIZE*3)	//Archive size
#define total_task_num	10		//Total task number
#define alpha_matde	0.1				//The rate of Transfer Learning Crossover
#define replace_rate	0.2		//Archive update rate (UR)
#define ro	0.8					//The attenuation coefficient
#define shrink_rate  0.8		//shrink rate of the local refined process
const int GA = 0;			//Whether or not use GA as STO

int task_num;					//The task number of current problem
int NVARS;						//Dimension of the current function being solved.
int NVARS_t[total_task_num];	//Dimension number of each task
double LBOUND[MAX_NVARS];		//Lower bound of the function being solved
double UBOUND[MAX_NVARS];		//Upper bound of the function being solved
const 	double PI = acos(-1.0);
int		generation;				//current generation
double possibility[total_task_num][total_task_num];	//Possibility table
double reward[total_task_num][total_task_num];	//Reward table
int	evals;	//Evaluation times
int job;	//Current run number


struct gene
{
	double x[MAX_NVARS];		//variable values
	double f[MAX_OBJS];			//objectives
};
//-------------------structure of chromosome.-------------------------
gene population[total_task_num][POPSIZE], newpopulation, mixpopulation[2 * POPSIZE];

//--------------------variables for result reporting-----------------
double fbest_value[total_task_num][MAX_GEN / RECORD_FRE + 1][RUNS];
// int TLC_record[total_task_num][MAX_GEN / RECORD_FRE][total_task_num];

//----------------for single-objective test------------------------
#define Single_Num	10
// int	successful_transfer[total_task_num][total_task_num][10];
double fbest[total_task_num];
//---------parameters for DE---------------------
#define LF_matde 0.1	
#define UF_matde 2
#define LCR_matde 0.1	
#define UCR_matde 0.9
double 	F;
double 	CR;

//-----------parameters for approximation measuring-----------
// double Cov[total_task_num][MAX_NVARS][MAX_NVARS];		//Covariance matrixs
// double Cov_Inv[total_task_num][MAX_NVARS][MAX_NVARS];	//Inverse covariance matrixs
double Cov_Det[total_task_num];							//Deteminents for covariance matrixs
// double Cov_Mean[total_task_num][MAX_NVARS];				//Mean vector for covariance matrixs	
double KLD[total_task_num];								//Similarities between a certain task to other tasks based on KL divergence
double avg_value[total_task_num][MAX_NVARS];			//Average values for each dimension
int k_archive_size[total_task_num];						//Current size of archive
// struct gene K_archive[total_task_num][K_ARCHIVE_SIZE];	//Archive
vector<vector<gene>> K_archive; 
// int trans_target;

// specified problem info 
#include "config.h"
#include "set_parameters.h"
#include "evaluator.h"
#include "EA.h"
#include "random.h"
#include "record.h"

// ProblemInfo problem_infos[total_task_num];
// BenchFuncEvaluator manytask_funs[total_task_num];
vector<unique_ptr<Evaluator>> manytask_funs;
Random random_;
string result_dir;
GA_CPU ga_cpu;
vector<Record> record_tasks;
vector<IslandInfo> island_infos;
vector<ProblemInfo> problem_infos;
vector<EAInfo> ea_infos;

int eval_init(int argc, char* argv[])
{
    Args args;
    IslandInfo island_info;
    ProblemInfo problem_info;
    NodeInfo node_info;
    EAInfo EA_info;
    int ret = SetParameters(island_info, problem_info, node_info, \
            EA_info, args, argc, argv);
    
    if(ret != 0)
    {
        fprintf(stderr,"Error: set parameters error.\n");
        exit(-1);
    }
    
    result_dir = args.results_dir + "/" + args.results_subdir;
    mkdirs(args.results_dir.c_str());
    mkdirs(result_dir.c_str());
    cout << "result dir " << result_dir << endl;
    
    CSimpleIni cfgs;
    fprintf(stderr, "Tasks define file: %s\n", args.tasks_def.c_str());
    
    cfgs.LoadFile(args.tasks_def.c_str());
	assert(args.total_tasks.size() == total_task_num);

    for(int i=0; i<args.total_tasks.size(); i++)
    {
        problem_info.task_id = args.total_tasks[i];
        problem_info.total_runs = args.total_runs;
        if(GetProblemInfo(cfgs, problem_info, args.use_unified_space) != 0) 
        {
            fprintf(stderr, "Error: get task info error from file %s.\n", args.tasks_def.c_str());
            return -1;
        }
		if (problem_info.problem_def == "Arm")
        {
            unique_ptr<Evaluator> eval_func(new ArmEvaluator());
            eval_func->Initialize(problem_info);
            manytask_funs.push_back(std::move(eval_func));
        }else {
            unique_ptr<Evaluator> eval_func(new BenchFuncEvaluator());
            eval_func->Initialize(problem_info);
            manytask_funs.push_back(std::move(eval_func));
        }
		
		island_info.results_dir = args.results_dir;
        island_info.island_ID = i;
        island_info.results_subdir = args.results_subdir;

        Record record = Record(node_info);
        record_tasks.push_back(record);

        island_infos.push_back(island_info);
        ea_infos.push_back(EA_info);
        problem_infos.push_back(problem_info);
    }
	CR = EA_info.CR;
	F = EA_info.F;
	fprintf(stderr, "!!! F=%.3f; CR=%.3f\n", F, CR);
	if (EA_info.STO == "GA")
	{
		// GA init
		ga_cpu = GA_CPU();
		ga_cpu.Initialize(island_info, problem_info, EA_info);
		printf("use GA as sto\n, alpha %.2f\n", alpha_matde);
	}else{
		printf("use DE as sto\n, alpha %.2f\n", alpha_matde);
	}
	
    return 0;
}

int eval_deinit()
{
    for(int i = 0; i < total_task_num; i++)
    {
        manytask_funs[i]->Uninitialize();
    }
	if (GA){
		ga_cpu.Uninitialize();
	}

    return 0;
}

double eval_manytask(double y[], int task_id)
{
	vector<real> elements(y, y + MAX_NVARS);
	double s = manytask_funs[task_id]->EvaluateFitness(elements);
	fbest[task_id] = min(fbest[task_id], s);
	return s;
}

double randval(double a, double b)
{
	return a + (b - a) * rand() / (double)RAND_MAX;
}
#include "util.h"
double min(double a, double b){
	if (a < b) return a;
	else return b;
}

//------------fitness calculation---------------------
void objective(int g, double x[MAX_NVARS], double f[])
{	
	f[0] = eval_manytask(x, g);
	evals += 1;
}

// #include <omp.h>
#include "Eigen/Dense"
vector<Eigen::MatrixXd> Cov_eigen;
vector<Eigen::MatrixXd> Inv_Cov_eigen;
bool calculated[total_task_num];

void set_calculated()
{
	for (int i = 0; i < total_task_num; i++)
	{
		calculated[i] = false;
	}
}

void get_Cov(int trans_target)
{
	if (calculated[trans_target])
	{
		return;
	}
	calculated[trans_target] = true;
	int i, j, l;
	// double start = get_wall_time();
	// #pragma omp parallel
    // {   
        // #pragma omp for
		for (i = 0; i < NVARS; i++){
			double s = 0;
			for (j = 0; j < k_archive_size[trans_target]; j++){
				s += K_archive[trans_target][j].x[i];
			}
			avg_value[trans_target][i] = s / k_archive_size[trans_target];
		}
	// }
	// #pragma omp parallel
    // {   
    //     #pragma omp for	
		for (i = 0; i < NVARS; i++){
			for (j = 0; j <= i; j++){
			double s = 0;
			for (l = 0; l < k_archive_size[trans_target]; l++)
				s += (K_archive[trans_target][l].x[i] - avg_value[trans_target][i]) * (K_archive[trans_target][l].x[j] - avg_value[trans_target][j]);
			// Cov[trans_target][i][j] = Cov[trans_target][j][i] = s / k_archive_size[trans_target];
			Cov_eigen[trans_target](i, j) = Cov_eigen[trans_target](j, i) = s / k_archive_size[trans_target];
			}
		}
	// }
	// printf("cov time %f\n", get_wall_time() - start);
}


void get_Cov_Det(int trans_target)
{
	if (calculated[trans_target])
	{
		return;
	}
	calculated[trans_target] = true;
	double det_eigen = Cov_eigen[trans_target].determinant();
	Cov_Det[trans_target] = det_eigen;
}

void get_Cov_Inv(int trans_target)
{
	if (calculated[trans_target])
	{
		return;
	}
	calculated[trans_target] = true;
	Inv_Cov_eigen[trans_target] = Cov_eigen[trans_target].inverse();
}

double get_Trace(int t1, int t2)
{
	int i, j, l;
	double sum;
	double ret = 0;
	for (i = 0; i < NVARS; i++){
		for (j = 0; j < NVARS; j++){
			if (i == j){
				for (l = 0; l < NVARS; l++)
				{
					ret += (Inv_Cov_eigen[t1](i, l) * Inv_Cov_eigen[t2](l, j)); 
				}
			}
		}
	}
	return ret;
}

double get_Mul(int t1, int t2)	
{
	// double start = get_wall_time();
	double a[MAX_NVARS];
	double sum;
	int i, j;
	// #pragma omp for
	for (i = 0; i < NVARS; i++){
		sum = 0;
		for (j = 0; j < NVARS; j++){
			// sum += (avg_value[t1][j] - avg_value[t2][j]) * Cov_Inv[t1][j][i];
			sum += (avg_value[t1][j] - avg_value[t2][j]) * Inv_Cov_eigen[t1](j, i);
		}
		a[i] = sum;
	}

	sum = 0;
	for (i = 0; i < NVARS; i++){
		sum += (avg_value[t1][i] - avg_value[t2][i]) * a[i];
	}

	return sum;
}

//Calculate the KLD between "task" task and the other tasks
void cal_KLD(int task)	
{
	int i, j;

	get_Cov(task);
	get_Cov_Det(task);
	get_Cov_Inv(task);
	// #pragma omp for
	for (i = 0; i < task_num; i++){
		double tr, u;
		double s1, s2;
		if (task == i) continue;
		NVARS = (NVARS_t[task] > NVARS_t[i] ? NVARS_t[i] : NVARS_t[task]);	//Pick the smaller dimension number to calculate KLD
		// get_Cov(task);
		get_Cov(i);
		// get_Cov_Det(task);
		// get_Cov_Inv(task);
		get_Cov_Det(i);
		get_Cov_Inv(i);

		tr = get_Trace(task, i);
		u = get_Mul(task, i);
		double det_i = Cov_Det[i];
		double det_task = Cov_Det[task];
		if (det_i < 1e-3) det_i = 0.001;
		if (det_task < 1e-3) det_task = 0.001;
		s1 = fabs(0.5 * (tr + u - NVARS + log(det_task / det_i)));
		
		tr = get_Trace(i, task);
		u = get_Mul(i, task);

		s2 = fabs(0.5 * (tr + u - NVARS + log(det_i / det_task)));
		
		KLD[i] = 0.5 * (s1 + s2);
	}
	NVARS = NVARS_t[task];
}

//Insert individual p into "task" archive
void put_k_archive(gene& p, int task)
{
	if (k_archive_size[task] < K_ARCHIVE_SIZE - 1)
		K_archive[task][k_archive_size[task]++] = p;
	else{
		int l = rand() % K_ARCHIVE_SIZE;
		K_archive[task][l] = p;
	}
}

int adaptive_choose(int task)
{
	int i;
	double sum = 0;
	int max = 0;
	double max_p = -1e10;
	double start;
	start = get_wall_time();
	cal_KLD(task);
	printf("calc KLD cost time %f\n", get_wall_time() - start);

	start = get_wall_time();
	//Update possibility table
	for (i = 0; i < task_num; i++){
		if (i == task) continue;
		possibility[task][i] = ro * possibility[task][i] + reward[task][i] / (1 + log(1 + KLD[i]));
		sum += possibility[task][i];
	}

	//roulette wheel selection
	double p = randval(0, 1);
	double s = 0;
	for (i = 0; i < task_num; i++){
		if (i == task) continue;
		s += possibility[task][i] / sum;
		if (s >= p) break;
	}
	// printf("task %d select %d as target\n", task, i);
	// printf("get selection cost time %f\n", get_wall_time() - start);
	return i;
}

void production(int task)
{
	int i, j, k, l;
	int r1;
	// double start = get_wall_time();
	if (randval(0, 1) > alpha_matde){   //perform the crossover and mutation within the subpopulation
		if(GA == 0)
		{
			// vector<gene> offspring;
			for (i = 0; i < POPSIZE; i++){
				// do{ r1 = rand() % POPSIZE; } while (i == r1);
				vector<int> r = random_.Permutate(POPSIZE, 3);
				// F = random_.RandRealUnif(LF_matde, UF_matde);
				// CR = random_.RandRealUnif(LCR_matde, UCR_matde);
				// printf("F %.2f CR %.2f\n", F, CR);
				// F = 0.5;
				// CR = 0.1;

				// k = rand() % NVARS;
				k = random_.RandIntUnif(0, NVARS-1);
				for (j = 0; j < MAX_NVARS; j++){
					// newpopulation.x[j] = population[task][i].x[j] + F * (population[task][r1].x[j] - population[task][i].x[j]);
					newpopulation.x[j] = population[task][r[0]].x[j] + F * (population[task][r[1]].x[j] - population[task][r[2]].x[j]);
					
					// if (newpopulation.x[j] > UBOUND[j]) newpopulation.x[j] = randval(population[task][i].x[j], UBOUND[j]);
					// if (newpopulation.x[j] < LBOUND[j]) newpopulation.x[j] = randval(LBOUND[j], population[task][i].x[j]);
					//new check bound
					while (newpopulation.x[j] < LBOUND[j] || newpopulation.x[j] > UBOUND[j])
					{
						if (newpopulation.x[j] < LBOUND[j])
						{
							newpopulation.x[j] = LBOUND[j] + (LBOUND[j] - newpopulation.x[j]);
						}
						if(newpopulation.x[j] > UBOUND[j])
						{
							newpopulation.x[j] = UBOUND[j] - (newpopulation.x[j] - UBOUND[j]);
						}
					}
					
					if (k == j || randval(0, 1) < CR){
					}
					else{
						newpopulation.x[j] = population[task][i].x[j];
					}
				}

				objective(task, newpopulation.x, newpopulation.f);
				// offspring.push_back(newpopulation);
				if (newpopulation.f[0] < population[task][i].f[0]){
					population[task][i] = newpopulation;
				}
			}
			// for(i = 0; i < POPSIZE; i++)
			// {
			// 	if (offspring[i].f[0] < population[task][i].f[0])
			// 	{
			// 		population[task][i] = offspring[i];	
			// 	}
			// }
		} else {
			// transform population to Population
			Population tmp_pop;
			for(i = 0; i < POPSIZE; i++)
			{
				Individual ind;
				for (j = 0; j < MAX_NVARS; j++)
				{
					ind.elements.push_back(population[task][i].x[j]);
				}
				ind.fitness_value = population[task][i].f[0];
				ind.skill_factor = -1;
				tmp_pop.push_back(ind);
			}
			// evolve
			int update_num = ga_cpu.Reproduce(tmp_pop, manytask_funs[task]);
			// transform Population to population
			for(i = 0; i < POPSIZE; i++)
			{
				for (j = 0; j < MAX_NVARS; j++)
				{
					population[task][i].x[j] = tmp_pop[i].elements[j];
				}
				population[task][i].f[0] = tmp_pop[i].fitness_value;
				fbest[task] = min(fbest[task], population[task][i].f[0]);
			}	
		}
	}
	else{
		//knowledge transferring�� 
		//perform the crossover and mutation cross different subpopulations		
		l = adaptive_choose(task);

		gene p = population[task][0];
		double s = 0;
		s = fbest[task];
		// TLC_record[task][generation / RECORD_FRE][l]++;
		for (i = 0; i < POPSIZE; i++){
			
			r1 = rand() % POPSIZE; 	
			double kt_CR = randval(LCR_matde, UCR_matde);
			k = rand() % NVARS;
			for (j = 0; j < MAX_NVARS; j++){
				if (k == j || randval(0, 1) < kt_CR){            //at least one dimension is replaced
					newpopulation.x[j] = population[l][r1].x[j];
				}
				else{
					newpopulation.x[j] = population[task][i].x[j];
				}
			}
			objective(task, newpopulation.x, newpopulation.f);

			if (newpopulation.f[0] < population[task][i].f[0]){		 
				population[task][i] = newpopulation;
				
			}
		}
		
		if (fbest[task] < s) {
			reward[task][l] /= shrink_rate;
			// successful_transfer[task][l][generation / 100]++;
		}
		else reward[task][l] *= shrink_rate;
	}
	// printf("produce one generation cost time %f\n", get_wall_time() - start);
}

//Parameter initialization
void initialized()
{
	int i, j, l;
	evals = 0;

	for (int i = 0; i < total_task_num; i++){
		NVARS_t[i] = MAX_NVARS;
		record_tasks[i].Initialize(island_infos[i], problem_infos[i], ea_infos[i]);
		Eigen::MatrixXd mat(MAX_NVARS, MAX_NVARS);
		Cov_eigen.push_back(mat);
		Inv_Cov_eigen.push_back(mat);
		K_archive.push_back(vector<gene>(K_ARCHIVE_SIZE, gene()));
	} 
	// NVARS_t[3] /= 2;
	for (i = 0; i < task_num; i++) fbest[i] = 1e20;
	for (i = 0; i < MAX_NVARS; i++){
		UBOUND[i] = 1;
		LBOUND[i] = 0;
	}

	memset(k_archive_size, 0, sizeof(k_archive_size));


	for (l = 0; l < task_num; l++){
		for (i = 0; i < POPSIZE; i++){
			for (j = 0; j < MAX_NVARS; j++){
				population[l][i].x[j] = random_.RandRealUnif(LBOUND[j], UBOUND[j]);
			}
			NVARS = NVARS_t[l];
			objective(l, population[l][i].x, population[l][i].f);
			put_k_archive(population[l][i], l);
		}
	}

	for (i = 0; i < task_num; i++){
		for (j = 0; j < task_num; j++){
			possibility[i][j] = 1.0 / (total_task_num - 1);
			reward[i][j] = 1;
		}
	}



}

void MaTDE()
{
	int i, j;
	initialized();
	generation = 0;
	//------------record the initial results---------------------
	for (i = 0; i < task_num; i++){
		 fbest_value[i][0][job] = fbest[i];
	}
	printf("generation:%d\n", generation);
	for (i = 0; i < task_num; i++){
		printf("fbest%d=%lf\n", i + 1, fbest_value[i][0][job]);
	}

	while (generation < MAX_GEN){
		double start = get_wall_time();
		//generate offspring based on the two populations, knowledge transfer will happen here.
		set_calculated();
		for (i = 0; i < task_num; i++){
				NVARS = NVARS_t[i];
				production(i);
		}
		//Update archive
		for (i = 0; i < task_num; i++) {
			for (j = 0; j < POPSIZE; j++) if (randval(0, 1) < replace_rate)
				put_k_archive(population[i][j], i);
		}

		//Print the current best
		if ((generation + 1) % record_tasks[0].RECORD_INTERVAL == 0
             || (generation == 0))
		{
			for (i = 0; i < task_num; i++){
					RecordInfo info;
					info.best_fitness = fbest[i];
					info.generation = generation+1;
					// info.elements = ind.elements;
					info.comm_time = 0;
					info.time = 0;
					record_tasks[i].RecordInfos(info);
					
					printf("run:%d\n", job);
					printf("generation:%d\t", generation + 1);
					printf("fbest%d=%lf\n", i + 1, fbest[i]);				
			}	
		}
		
		generation++;
		fprintf(stderr, "one gen cost time %f\n", get_wall_time() - start);
	}
	cerr << "start uninitialize " << endl;
	for (i = 0; i < task_num; i++)
	{
		record_tasks[i].FlushInfos();
		record_tasks[i].Uninitialize();
	}
	cerr << "finish one run " << endl;	

}

int main(int argc, char* argv[])
{
	double duration;
	double total = 0;
	clock_t start;
	int k = 0;
	//Set task number 
	task_num = total_task_num;
	start = clock();
    eval_init(argc, argv);
	for (job = 0; job < RUNS; job++)
	{
		double time_start = get_wall_time();
		printf("\njob = %d\n", job);
		srand((job+1)*10000);
		// srand(job);
		MaTDE();
		cout << "one run cost time " << get_wall_time() - time_start << endl; 
	}
	cout << "total time cost: " << (double)(clock() - start) / CLOCKS_PER_SEC << " s for " << RUNS << " runs" << endl;
	// final_report();
    eval_deinit();
	return 0;
}



