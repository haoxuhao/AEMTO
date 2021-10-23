#include "config.h"
#include "set_parameters.h"
#include "evaluator.h"
#include "EA.h"
#include "random.h"
#include "util.h"
#include "record.h"

using namespace std;

int	evals = 0;	//evaluation times
int run_id;	
int ntasks = 1;
vector<unique_ptr<Evaluator>> task_evals;
vector<ProblemInfo> problem_infos;
vector<Population> pop_tasks;
vector<Record> record_tasks;
Random random_;
Args args;
EAInfo EA_info;
unique_ptr<EA> EA_solver;

/*SBO matrixs*/
vector<vector<Real>> M, N, C, O, P, A, R;

int global_init(int argc, char* argv[])
{
    int ret = SetParameters(argc, argv, args, EA_info);
    if(ret != 0)
    {
        fprintf(stderr,"Error: set parameters error.\n");
        exit(-1);
    }
    mkdirs(args.results_dir.c_str());
    args.total_runs = args.total_runs;
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

    ntasks = (int)args.total_tasks.size();
    fprintf(stderr, "=================== INFO ============\n");
    fprintf(stderr, "total tasks = %d; total runs = %d\n", ntasks, args.total_runs);
    fprintf(stderr, "pop_sizexm = %dx%d\n", args.popsize, ntasks);
    fprintf(stderr, "args.Gmax = %d\n", args.Gmax);
    fprintf(stderr, "results dir = %s\n", args.results_dir.c_str());
    fprintf(stderr, "EA solver = %s.\n", EA_info.STO.c_str());
    fprintf(stderr, "MTO = %d\n", args.MTO);

    if (EA_info.STO == "GA")
    {
        EA_solver.reset(new GA(problem_infos[0], EA_info));
    }else if (EA_info.STO == "DE")
    {
        EA_solver.reset(new DE(problem_infos[0], EA_info));
    }
    else{
        fprintf(stderr, "Error no EA solver found: %s.\n", EA_info.STO.c_str());
        exit(-1);
    }
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
    }else if (rate_i <= bene_rate && rate_j <= harm_rate)
    {
        O[i][j]++;
    }else if (rate_i <= bene_rate && rate_j > harm_rate)
    {
        P[i][j]++;
    }
    else if (rate_i <= harm_rate && rate_j <= harm_rate)
    {
        N[i][j]++;
    }else if (rate_i <= harm_rate && rate_j > harm_rate)
    {
        A[i][j]++;
    }else if (rate_i > harm_rate && rate_j > harm_rate)
    {
        C[i][j]++;
    }else {
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
        pop_tasks[i].resize(args.popsize, Individual(args.UDim));
        EA_solver->InitializePopulation(pop_tasks[i], task_evals[i]);
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
    if(args.MTO)
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
	int generation = 0;
    Real bestf = 0;
    int rank_i, rank_j, j;
    Real origin_fitness;

	while (generation < args.Gmax){
        vector<Population> offsprings;
		for (int i = 0; i < ntasks; i++){
            Population offsp = EA_solver->Variation(pop_tasks[i]);
            offsprings.emplace_back(offsp);
		}
        for (int i = 0; i < ntasks; i++)
        {
            if (args.MTO)
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
            
            offsprings[i] = EA_solver->EvaluatePop(offsprings[i], task_evals[i]);
            EvolveRewards rewards;
            pop_tasks[i] = EA_solver->Survival(pop_tasks[i], offsprings[i], rewards);
            offsprings[i] = EA_solver->SortPop(offsprings[i]);
        }
        if (args.MTO){
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
                        origin_fitness = task_evals[j]->EvaluateFitness(offsprings[i][k].elements);
                        rank_j = rank_in_pop(offsprings[j], origin_fitness);
                        undefined_rule += update_SB_matrix(i, j,
                            rank_i / (Real)lambda, rank_j / (Real)lambda);
                    }
                }
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
                fprintf(stderr, "task %d; runs %d/%d; gens %d/%d; bestf %.12f\n", i+1, run_id+1, args.total_runs, generation+1, args.Gmax, bestf);
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
                // if (i == 0)
                //     fprintf(stdout, "task %d selection probability; runs %d/%d; gens %d/%d: [%s].\n",
                //         i+1, run_id+1, args.total_runs, generation+1, args.Gmax, ss.str().c_str());

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
        fprintf(stderr, "task %d run_id %d, final results: [%s %.12f] \n", i+1, run_id+1, ss.str().c_str(), bestf);
        record_tasks[i].FlushInfos(run_id);
    }
    
    uninitialized();

}
int main(int argc, char* argv[])
{
	double duration;
	double total = 0;
	auto start = get_wall_time();

    global_init(argc, argv);
	for (run_id = 0; run_id < args.total_runs; run_id++)
	{
        double time_s = get_wall_time();
		srand((run_id+1)*10000);
		SBO();
        cout << "one run cost time " << get_wall_time() - time_s << endl;
	}
	cout << "total time cost: " << get_wall_time() - start
         << " seconds for " << args.total_runs << " runs" << endl;
	return 0;
}
