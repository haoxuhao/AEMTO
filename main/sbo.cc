#include "config.h"
#include "set_parameters.h"
#include "evaluator.h"
#include "EA.h"
#include "random.h"
#include "util.h"
#include "record.h"

using namespace std;

bool MTO = true;// 0 for no transfer; 1 for transfer

int generation;	//current generation
int	evals;	//Evaluation times
int job;	//Current run number
int ntasks;
int RUNS = 1;
int MAX_GENS = 1000;
long MAX_EVALS = 100000;

vector<unique_ptr<Evaluator>> manytask_funs;
vector<IslandInfo> island_infos;
vector<ProblemInfo> problem_infos;
vector<EAInfo> ea_infos;
vector<Population> pop_tasks;
vector<Record> record_tasks;
Random random_;
string result_dir;
Args args;

EA *EA_solver;

/*SBO matrixs*/
vector<vector<Real>> M, N, C, O, P, A, R;

int global_init(int argc, char* argv[])
{
    IslandInfo island_info;
    ProblemInfo problem_info;
    EAInfo EA_info;
    int ret = SetParameters(island_info, problem_info,  \
            EA_info, args, argc, argv);
    
    if(ret != 0)
    {
        fprintf(stderr,"Error: set parameters error.\n");
        exit(-1);
    }
    
    result_dir = args.results_dir + "/" + args.results_subdir;
    mkdirs(args.results_dir.c_str());
    mkdirs(result_dir.c_str());

    RUNS = args.total_runs;

    for(int i=0; i<args.total_tasks.size(); i++)
    {
        problem_info.task_id = args.total_tasks[i];
        problem_info.total_runs = args.total_runs;
        if(GetProblemInfo(args, problem_info) != 0) 
        {
            fprintf(stderr, "Error: task %d, get task info error from.\n", problem_info.task_id);
            return -1;
        }
        if (problem_info.problem_def == "Arm")
        {
            unique_ptr<Evaluator> eval_func(new ArmEvaluator(problem_info));
            manytask_funs.push_back(std::move(eval_func));
        }else {
            unique_ptr<Evaluator> eval_func(new BenchFuncEvaluator(problem_info));
            manytask_funs.push_back(std::move(eval_func));
        }
        Record record = Record(args, args.total_tasks[i]);
        record_tasks.push_back(record);

        island_infos.push_back(island_info);
        ea_infos.push_back(EA_info);
        problem_infos.push_back(problem_info);
    }

    MAX_GENS = args.G_max;
    ntasks = (int)args.total_tasks.size();
    fprintf(stderr, "=================== INFO ============\n");
    fprintf(stderr, "total tasks = %d; total runs = %d\n", ntasks, RUNS);
    fprintf(stderr, "pop_sizexm = %dx%d\n", island_info.island_size, ntasks);
    fprintf(stderr, "MAX_GENS = %d\n", MAX_GENS);
    fprintf(stderr, "results dir = %s\n", result_dir.c_str());
    fprintf(stderr, "EA solver = %s.\n", EA_info.STO.c_str());
    fprintf(stderr, "MTO = %d\n", MTO);

    if (EA_info.STO == "GA")
	{
		EA_solver = new GA_CPU();
        EA_solver->Initialize(island_info, problem_info, EA_info);
        
	}else if (EA_info.STO == "DE")
    {
        EA_solver = new DE_CPU();
        EA_solver->Initialize(island_info, problem_info, EA_info);
    }
     else{
        fprintf(stderr, "Error no EA solver found: %s.\n", EA_info.STO.c_str());
        exit(-1);
    }
    return 0;
}

int global_deinit()
{
    delete EA_solver;
    return 0;
}

int update_R()
{
    int n = R.size();
    int pos, neg, neu;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                pos = M[i][j] + O[i][j] + P[i][j];
                neg = A[i][j] + C[i][j];
                neu = N[i][j];
                R[i][j] = pos / (Real)(pos + neg + neu);
            }
        }
    }
    return 0;
}

int update_SB_matrix(int i, int j, Real rate_i, Real rate_j)
{
    Real bene_rate = 0.25; //beneficial rate
    Real harm_rate = 0.5; // harmful rate
    if (rate_i <= bene_rate && rate_j <= bene_rate)
    {
        M[i][j]++;
        // fprintf(stderr, "(rate_i, rate_j)= (%.4f, %.4f), update M\n", rate_i, rate_j);
    }else if (rate_i <= bene_rate && rate_j <= harm_rate)
    {
        O[i][j]++;
        // fprintf(stderr, "(rate_i, rate_j)=(%.4f, %.4f), update O\n", rate_i, rate_j);
    }else if (rate_i <= bene_rate && rate_j > harm_rate)
    {
        P[i][j]++;
        // fprintf(stderr, "(rate_i, rate_j)=(%.4f, %.4f), update P\n", rate_i, rate_j);
    }
    else if (rate_i <= harm_rate && rate_j <= harm_rate)
    {
        N[i][j]++;
        // fprintf(stderr, "(rate_i, rate_j)=(%.4f, %.4f), update N\n", rate_i, rate_j);
    }else if (rate_i <= harm_rate && rate_j > harm_rate)
    {
        A[i][j]++;
        // fprintf(stderr, "(rate_i, rate_j)=(%.4f, %.4f), update A\n", rate_i, rate_j);
    }else if (rate_i > harm_rate && rate_j > harm_rate)
    {
        C[i][j]++;
        // fprintf(stderr, "(rate_i, rate_j)=(%.4f, %.4f), update C\n", rate_i, rate_j);
    }else {
        // fprintf(stderr, "undefined rules for rate (i, j) = (%f, %f)\n", rate_i, rate_j);
        return 1;
    }
    return 0;
}

void production(int task)
{

}

int argmax(vector<Real> x, int skip_index)
{
    int max_id = -1;
    Real max_val = -1;
    for (int i = 0; i < x.size(); i++)
    {
        if (i != skip_index)
        {
            if (x[i] > max_val)
            {
                max_id = i;
                max_val = x[i];
            }
        }
    }
    return max_id;
}

int rank_in_pop(Population &p, Real value)
{
    int l = 0, r = p.size() - 1, mid;
    if(l > r)
    {
        return -1;
    }
    while (l <= r)
    {
        mid = l + (r - l) / 2;
        if (p[mid].fitness_value < value)
        {
            l = mid + 1;
        }
        else if (p[mid].fitness_value > value)
        {
            r = mid - 1;
        }
        else
        {
            return mid;
        }
    }
    return l;
}

//initialization
void initialized()
{
	evals = 0;
    pop_tasks.resize(ntasks, Population());
    for (int i = 0; i < ntasks; i++)
    {
        EA_solver->InitializePopulation(pop_tasks[i], manytask_funs[i]);
        evals += pop_tasks[i].size();
    }
    // SBO matrix init
    for(int i = 0; i < ntasks; i++)
    {
        M.emplace_back(vector<Real>(ntasks, 1));
        N.emplace_back(vector<Real>(ntasks, 1));
        C.emplace_back(vector<Real>(ntasks, 1));
        O.emplace_back(vector<Real>(ntasks, 1));
        P.emplace_back(vector<Real>(ntasks, 1));
        A.emplace_back(vector<Real>(ntasks, 1));
        R.emplace_back(vector<Real>(ntasks, 0));
    }
    if(MTO)
        update_R();
}

void uninitialized()
{
    pop_tasks.clear();
    M.clear(); N.clear(); C.clear();
    O.clear(); P.clear(); A.clear();
    R.clear();
    for(int i = 0; i < ntasks; i++)
    {
        record_tasks[i].Clear();
    }
}

void SBO()
{
	initialized();
	generation = 0;
    Real bestf = 0;
    int rank_i, rank_j, j;
    Real origin_fitness;

	while (generation < MAX_GENS){
        vector<Population> offsprings;
		for (int i = 0; i < ntasks; i++){
            Population offsp = EA_solver->Variation(pop_tasks[i]);
            offsprings.emplace_back(offsp);
		}
        for (int i = 0; i < ntasks; i++)
        {
            if (MTO)
            {
                j = argmax(R[i], i);
                Real Ri = R[i][j];
                if (random_.RandRealUnif(0, 1) < Ri)
                {
                    int lambda = offsprings[i].size();
                    int Si = Ri * lambda;
                    int src_index, dst_index;
                    for (int k = 0; k < Si; k++)
                    {
                        src_index = k;
                        dst_index = lambda - Si - 1 + k;
                        offsprings[i][dst_index] = offsprings[j][src_index];
                        offsprings[i][dst_index].skill_factor = j; 
                    }
                }
            }
            
            offsprings[i] = EA_solver->EvaluatePop(offsprings[i], manytask_funs[i]);
            EvolveRewards rewards;
            pop_tasks[i] = EA_solver->Survival(pop_tasks[i], offsprings[i], rewards);
            offsprings[i] = EA_solver->SortPop(offsprings[i]);
        }
        if (MTO){
            // update M,N...
            for (int i = 0; i < ntasks; i++)
            {
                int lambda = offsprings[i].size();
                int total_transfered = 0;
                int undefined_rule = 0;
                for (int k = 0; k < lambda; k++)
                {
                    j = offsprings[i][k].skill_factor;
                    if (j != -1)
                    {
                        total_transfered ++;
                        rank_i = k;
                        origin_fitness = manytask_funs[j]->EvaluateFitness(offsprings[i][k].elements);
                        rank_j = rank_in_pop(offsprings[j], origin_fitness);
                        // printf("run id %d, gens %d, island %d, curr FV %.6f; rank i %d, rank j(%d) %d, FV of j %.6f\n", 
                        //     job+1, generation+1, i, offsprings[i][k].fitness_value, rank_i,
                        //     j, rank_j, origin_fitness);
                        undefined_rule += update_SB_matrix(i, j,
                            rank_i / (Real)lambda, rank_j / (Real)lambda);
                    }
                }
                // if (total_transfered != 0)
                //     printf("run id %d, gens %d, island %d: transferred %d, undefined rule %d, undefined rate %.4f\n",
                //         job+1, generation+1, i, total_transfered, undefined_rule, undefined_rule / (Real)(total_transfered + 1e-12));
            }
            // update R
            update_R();
        }
        
		//Print the current best
		if ((generation + 1) % args.record_interval == 0 || (generation == 0))
        {
			for (int i = 0; i < ntasks; i++)
            {
                Individual ind = EA_solver->FindBestIndividual(pop_tasks[i]);
                bestf = ind.fitness_value; 
                fprintf(stderr, "island %d; runs %d/%d; gens %d/%d; bestf %.12f\n", i, job+1, RUNS, generation+1, MAX_GENS, bestf);
                stringstream ss;
                for(int k = 0; k < ntasks; k++)
                {
                    if(k != ntasks - 1)
                    {
                        ss << R[i][k] << ", ";
                    }else
                    {
                        ss << R[i][k];
                    }
                }
                if (i == 0)
                    fprintf(stdout, "island %d selection probability; runs %d/%d; gens %d/%d: [%s].\n",
                        i, job+1, RUNS, generation+1, MAX_GENS, ss.str().c_str());

                RecordInfo info;
                info.best_fitness = bestf;
                info.generation = generation+1;
                // info.elements = ind.elements;
                record_tasks[i].RecordInfos(info);
            }
        }
        generation++;
	}
    for (int i = 0; i < ntasks; i++)
    {
        Individual ind = EA_solver->FindBestIndividual(pop_tasks[i]);
        bestf = ind.fitness_value;
        stringstream ss;
        for (auto e : ind.elements)
        {
            ss << e << ", ";
        }
        fprintf(stderr, "island %d run_id %d, final results: [%s %.12f] \n", i, job+1, ss.str().c_str(), bestf);
        record_tasks[i].FlushInfos(job);
    }
    
    uninitialized();

}
int main(int argc, char* argv[])
{
	double duration;
	double total = 0;
	clock_t start = clock();

    global_init(argc, argv);
	for (job = 0; job < RUNS; job++)
	{
        double time_s = get_wall_time();
		srand((job+1)*10000);
		SBO();
        cout << "one run cost time " << get_wall_time() - time_s << endl;
	}
	cout << "total time cost: " << (double)(clock() - start) / CLOCKS_PER_SEC << " s for " << RUNS << " RUNS" << endl;
    global_deinit();
	return 0;
}
