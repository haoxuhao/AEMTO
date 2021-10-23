#include <iostream>
#include <vector>
#include <ctime>
#include <sstream>
#include "config.h"
#include "set_parameters.h"
#include "evaluator.h"
#include "EA.h"
#include "random.h"
#include "record.h"
#include "Eigen/Dense"
#include "util.h"

using namespace std;

struct gene
{
	gene(int dim) {
		x.resize(dim);
	};
	vector<double> x;		//variable values
	double f;			
};


int	MAX_NVARS = 50;		//The maximum dimension of the tasks.
int MAX_GEN = 1000;	//Maximum generation
int K_ARCHIVE_SIZE = 0; //	//Archive size
int total_task_num = 10;

double alpha = 0.1;	//The rate of Transfer Learning Crossover
double replace_rate	= 0.2;	//Archive update rate (UR)
double ro = 0.8;			//The attenuation coefficient
double shrink_rate = 0.8;   //shrink rate of the local refined process

int NVARS;						//Dimension of the current function being solved.
vector<int> NVARS_t;
double LBOUND = 0;
double UBOUND = 1;
const double PI = acos(-1.0);
int	generation;			
vector<vector<double>> possibility;
vector<vector<double>> reward;
vector<double> fbest;
int	evals;	
int run_id;
double F;
double CR;

vector<vector<gene>> population;

vector<double> Cov_Det;//Deteminents for covariance matrixs
vector<double> KLD;//Similarities between a certain task to other tasks based on KL divergence
vector<vector<double>> avg_value;//Average values for each dimension
vector<int> k_archive_size;//Current size of archive					
vector<vector<gene> > K_archive; 
vector<bool> calculated;

vector<Eigen::MatrixXd> Cov_eigen;
vector<Eigen::MatrixXd> Inv_Cov_eigen;

vector<unique_ptr<Evaluator>> task_evals;
Random random_;
vector<Record> record_tasks;
vector<ProblemInfo> problem_infos;
Args args;
EAInfo EA_info;


int eval_init(int argc, char* argv[])
{
    ProblemInfo problem_info;
    
    int ret = SetParameters(argc, argv, args, EA_info);
    
    if(ret != 0)
    {
        fprintf(stderr,"Error: set parameters error.\n");
        exit(-1);
    }
    mkdirs(args.results_dir.c_str());
    
	problem_infos = GetProblemInfos(args);
    for (int i  = 0; i < args.total_tasks.size(); i++) {
        if (args.problem_set == "Arm")
        {
            unique_ptr<Evaluator> eval_func(new ArmEvaluator(problem_infos[i]));
            task_evals.push_back(std::move(eval_func));
        } else {
            unique_ptr<Evaluator> eval_func(new BenchFuncEvaluator(problem_infos[i]));
            task_evals.push_back(std::move(eval_func));
        }
        record_tasks.emplace_back(args, args.total_tasks[i]);
        problem_infos.push_back(problem_infos[i]);
    }

	CR = EA_info.CR;
	F = EA_info.F;
	total_task_num = args.total_tasks.size();
	fbest.resize(total_task_num);
	calculated.resize(total_task_num);
	Cov_Det.resize(total_task_num);//Deteminents for covariance matrixs
	KLD.resize(total_task_num);//Similarities between a certain task to other tasks based on KL divergence
	avg_value.resize(total_task_num, vector<double>(MAX_NVARS));//Average values for each dimension
	k_archive_size.resize(total_task_num);//Current size of archive					
	NVARS_t.resize(total_task_num, 0);
	K_ARCHIVE_SIZE = 3*args.popsize;
	MAX_NVARS = args.UDim;
	population.resize(total_task_num, vector<gene>(args.popsize, gene(MAX_NVARS)));
	possibility.resize(total_task_num, vector<double>(total_task_num, 0.0));
	reward.resize(total_task_num, vector<double>(total_task_num, 0.0));
	
    return 0;
}

double randval(double a, double b)
{
	return a + (b - a) * rand() / (double)RAND_MAX;
}

double objective(int task_id, vector<double> &x)
{	
	double f = task_evals[task_id]->EvaluateFitness(x);
	fbest[task_id] = std::min(fbest[task_id], f);
	evals += 1;
	return f;
}

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
	for (i = 0; i < NVARS; i++){
		double s = 0;
		for (j = 0; j < k_archive_size[trans_target]; j++){
			s += K_archive[trans_target][j].x[i];
		}
		avg_value[trans_target][i] = s / k_archive_size[trans_target];
	}
	for (i = 0; i < NVARS; i++){
		for (j = 0; j <= i; j++){
		double s = 0;
		for (l = 0; l < k_archive_size[trans_target]; l++)
			s += (K_archive[trans_target][l].x[i] - avg_value[trans_target][i]) * (K_archive[trans_target][l].x[j] - avg_value[trans_target][j]);
		Cov_eigen[trans_target](i, j) = Cov_eigen[trans_target](j, i) = s / k_archive_size[trans_target];
		}
	}
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
	double a[MAX_NVARS];
	double sum;
	int i, j;
	for (i = 0; i < NVARS; i++){
		sum = 0;
		for (j = 0; j < NVARS; j++){
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
	for (i = 0; i < total_task_num; i++){
		double tr, u;
		double s1, s2;
		if (task == i) continue;
		NVARS = (NVARS_t[task] > NVARS_t[i] ? NVARS_t[i] : NVARS_t[task]);	//Pick the smaller dimension number to calculate KLD
		get_Cov(i);
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
	cal_KLD(task);

	//update possibility table
	for (i = 0; i < args.total_tasks.size(); i++){
		if (i == task) continue;
		possibility[task][i] = ro * possibility[task][i] + reward[task][i] / (1 + log(1 + KLD[i]));
		sum += possibility[task][i];
	}

	//roulette wheel selection
	double p = randval(0, 1);
	double s = 0;
	for (i = 0; i < args.total_tasks.size(); i++){
		if (i == task) continue;
		s += possibility[task][i] / sum;
		if (s >= p) break;
	}
	return i;
}

void production(int task)
{
	int i, j, k, l;
	int r1;
	if (randval(0, 1) > alpha && args.MTO){ 
		for (i = 0; i < args.popsize; i++){
			vector<int> r = random_.Permutate(args.popsize, 3);
			k = random_.RandIntUnif(0, NVARS-1);
			gene newgene(args.UDim);
			for (j = 0; j < MAX_NVARS; j++){
				newgene.x[j] = population[task][r[0]].x[j] + F * (population[task][r[1]].x[j] - population[task][r[2]].x[j]);
				while (newgene.x[j] < LBOUND || newgene.x[j] > UBOUND)
				{
					if (newgene.x[j] < LBOUND)
					{
						newgene.x[j] = LBOUND + (LBOUND - newgene.x[j]);
					}
					if(newgene.x[j] > UBOUND)
					{
						newgene.x[j] = UBOUND - (newgene.x[j] - UBOUND);
					}
				}
				if (k == j || randval(0, 1) < CR){
				}
				else{
					newgene.x[j] = population[task][i].x[j];
				}
			}

			newgene.f = objective(task, newgene.x);
			if (newgene.f < population[task][i].f){
				population[task][i] = newgene;
			}
		}
	}
	else
	{
		l = adaptive_choose(task);
		gene p = population[task][0];
		double s = 0;
		s = fbest[task];
		static const double LCR = 0.1;
		static const double UCR = 0.9;
		for (i = 0; i < args.popsize; i++){
			r1 = rand() % args.popsize; 	
			double kt_CR = randval(LCR, UCR);
			k = rand() % NVARS;
			gene newgene(args.UDim);
			for (j = 0; j < MAX_NVARS; j++){
				if (k == j || randval(0, 1) < kt_CR){           
					newgene.x[j] = population[l][r1].x[j];
				}
				else{
					newgene.x[j] = population[task][i].x[j];
				}
			}
			newgene.f = objective(task, newgene.x);

			if (newgene.f < population[task][i].f){		 
				population[task][i] = newgene;
			}
		}
		
		if (fbest[task] < s) {
			reward[task][l] /= shrink_rate;
		}
		else reward[task][l] *= shrink_rate;
	}
}


void initialized()
{
	int i, j, l;
	evals = 0;

	for (int i = 0; i < total_task_num; i++){
		NVARS_t[i] = problem_infos[i].calc_dim;
		Eigen::MatrixXd mat(MAX_NVARS, MAX_NVARS);
		Cov_eigen.push_back(mat);
		Inv_Cov_eigen.push_back(mat);
		K_archive.push_back(vector<gene>(K_ARCHIVE_SIZE, gene(MAX_NVARS)));
	} 
	for (i = 0; i < total_task_num; i++) fbest[i] = 1e20;

	k_archive_size.assign(k_archive_size.size(), 0);


	for (l = 0; l < total_task_num; l++){
		for (i = 0; i < args.popsize; i++){
			for (j = 0; j < MAX_NVARS; j++){
				population[l][i].x[j] = random_.RandRealUnif(LBOUND, UBOUND);
			}
			// NVARS = NVARS_t[l];
			population[l][i].f = objective(l, population[l][i].x);
			put_k_archive(population[l][i], l);
		}
	}

	for (i = 0; i < total_task_num; i++){
		for (j = 0; j < total_task_num; j++){
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
	for (i = 0; i < args.total_tasks.size(); i++){
		printf("task %d init fbest %lf\n", i + 1, fbest[i]);
	}

	while (generation < args.Gmax){
		set_calculated();
		for (i = 0; i < total_task_num; i++){
				NVARS = NVARS_t[i];
				production(i);
		}

		for (i = 0; i < total_task_num; i++) {
			for (j = 0; j < args.popsize; j++) if (randval(0, 1) < replace_rate)
				put_k_archive(population[i][j], i);
		}

		if ((generation + 1) % args.record_interval == 0
             || (generation == 0))
		{
			for (i = 0; i < args.total_tasks.size(); i++){
				RecordInfo info;
				info.best_fitness = fbest[i];
				info.generation = generation+1;
				record_tasks[i].RecordInfos(info);
				fprintf(stderr, "task %d; runs %d/%d; gens %d/%d; "
                                "bestf %.12f\n",
                                 i + 1, run_id + 1, args.total_runs, generation + 1, 
								 args.Gmax, fbest[i]);	
									
			}	
		}
		
		generation++;
	}
	for (int i = 0; i < args.total_tasks.size(); i++)
    {
        // auto ind = EA_solver->FindBestIndividual(pop_tasks[i]);
        Real bestf = fbest[i];
        // stringstream ss;
        // for (auto e : ind.elements) {
        //     ss << e << ", ";
        // }
        // fprintf(stderr, "task %d run_id %d, final solution (in unified space): [%s]; "
        //                 "final bestf: %.12f \n", 
        //                 i+1, run_id + 1, ss.str().c_str(), 
        //                 bestf);
        record_tasks[i].FlushInfos(run_id);
		record_tasks[i].Clear();
    }
}

int main(int argc, char* argv[])
{
	clock_t start = clock();
    eval_init(argc, argv);
	for (run_id = 0; run_id < args.total_runs; run_id++)
	{
		double time_start = get_wall_time();
		printf("\nrun_id = %d\n", run_id);
		srand((run_id+1)*10000);
		MaTDE();
		cout << "one run cost time " << get_wall_time() - time_start << endl; 
	}
	cout << "total time cost: " << (double)(clock() - start) / CLOCKS_PER_SEC 
		 << " s for " << args.total_runs << " runs" << endl;
	return 0;
}



