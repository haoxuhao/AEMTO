#include <sstream>
#include <algorithm>
#include "config.h"
#include "util.h"
#include "selection.h"
#include "record.h"
#include "random.h"
#include "EA.h"
#include "set_parameters.h"

using namespace std;

bool MTO = true;
const real eps = 1e-12;
int	evals;	//Evaluation times
int job;	//Current run number
int ntasks;
int RUNS = 1;
int MAX_GENS = 1000;
int MAX_FES = 0;
real p_tsf_lb = 0.05; // lower bound of transfer probability
real p_tsf_ub = 0.70; // upper ...
real alpha=0.3;
real pbase=0.2;
vector<real> tsf_probs;
vector<real> rewards_self;
vector<real> rewards_other;
vector<int> tsf_cnt;

vector<unique_ptr<Evaluator> > task_evals; // evaluator of each task
vector<IslandInfo> island_infos;
vector<ProblemInfo> problem_infos;
vector<EAInfo> ea_infos;
unique_ptr<EA> EA_solver; // EA solver
vector<Selection> task_sels; // selection module for each task

vector<Population> pop_tasks;
vector<Record> record_tasks;
Random rand_;
string result_dir;

int global_init(int argc, char* argv[])
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

    island_info.island_num = args.total_tasks.size();
    island_info.task_ids.assign(args.total_tasks.begin(), args.total_tasks.end());
    RUNS = args.total_runs;
    if(island_info.export_prob == 0)
    {
        MTO = false;
    }

    CSimpleIni cfgs;
    fprintf(stderr, "Tasks define file: %s\n", args.tasks_def.c_str());
    cfgs.LoadFile(args.tasks_def.c_str());
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
            task_evals.push_back(std::move(eval_func));
        } else {
            unique_ptr<Evaluator> eval_func(new BenchFuncEvaluator());
            eval_func->Initialize(problem_info);
            task_evals.push_back(std::move(eval_func));
        }
        island_info.results_dir = args.results_dir;
        island_info.island_ID = i;
        island_info.results_subdir = args.results_subdir;

        Record record = Record(node_info);
        record_tasks.push_back(record);

        island_infos.push_back(island_info);
        ea_infos.push_back(EA_info);
        problem_infos.push_back(problem_info);
        task_sels.push_back(Selection());
        
    }
    if (EA_info.STO == "GA")
    {
        EA_solver.reset(new GA_CPU());
        EA_solver->Initialize(island_info, problem_info, EA_info);
    }else if (EA_info.STO == "DE")
    {
        EA_solver.reset(new DE_CPU());
        EA_solver->Initialize(island_info, problem_info, EA_info);
    }
    else{
        fprintf(stderr, "Error no EA solver found: %s.\n", EA_info.STO.c_str());
        exit(-1);
    }
    MAX_FES = island_info.run_param.FEs;
    MAX_GENS = MAX_FES / island_info.island_size;
    ntasks = (int)args.total_tasks.size();
    fprintf(stderr, "=================== INFO ============\n");
    fprintf(stderr, "total tasks = %d; total runs = %d\n", ntasks, RUNS);
    fprintf(stderr, "pop_size x m = %dx%d\n", island_info.island_size, ntasks);
    fprintf(stderr, "MAX_EVALS = %ldx%d; MAX_GENS = %d\n", MAX_FES, ntasks, MAX_GENS);
    fprintf(stderr, "results dir = %s\n", result_dir.c_str());
    fprintf(stderr, "EA solver = %s.\n", EA_info.STO.c_str());
    fprintf(stderr, "problem set = %s.\n", problem_info.problem_def.c_str());
    fprintf(stderr, "MTO = %d\n", MTO);

    return 0;
}

int global_deinit()
{
    for(int i = 0; i < ntasks; i++) {
        task_evals[i]->Uninitialize();
    }
	EA_solver->Uninitialize();
    return 0;
}

Population select_from_others(int current_tid)
{
    Population immigrants;
    const IslandInfo &current = island_infos[current_tid];
    if (current.island_num == 1) {
        return immigrants;
    }
    unordered_map<int, int> selections = task_sels[current_tid].get_import_selections();
    for (auto ele : selections)
    {
        int other_task_id = ele.first;
        int num_sel = ele.second;
        vector<real> fitnesses;
        const Population &pop_other = pop_tasks[other_task_id];
        for(const auto & e : pop_other) {
            fitnesses.push_back(e.fitness_value);
        }
        vector<int> indices = rand_.roulette_sample(fitnesses, num_sel);
        for (const auto &i : indices)
        {
            Individual ind = pop_other.at(i);
            ind.skill_factor = other_task_id;
            immigrants.emplace_back(ind);
        }
    }
    return immigrants;
}

real Reuse(int task_id, Population &pop, Population &other_pop, 
        unordered_map<int, int> &success_insert_table)
{
    int other_pop_size = other_pop.size();
    if(other_pop_size == 0) return 0.0;
    vector<int> rand_indexs = rand_.Permutate(other_pop_size, other_pop_size);
    int N = pop.size();
    real update_num = 0;
    for (int k = 0; k < N; k++)
    {
        Individual mu = other_pop.at(rand_indexs[k % other_pop_size]);
        Individual x = pop.at(k);
        real cr = rand_.RandRealUnif(ea_infos[task_id].LKTCR, ea_infos[task_id].UKTCR);
        Individual c = binomial_crossover(x, mu, cr);
        c.fitness_value = task_evals[task_id]->EvaluateFitness(c.elements);
        if (c.fitness_value < x.fitness_value)
        {
            pop[k] = c;
            pop[k].skill_factor = -1;
            success_insert_table[mu.skill_factor]++;
            update_num++;
        }
    }
    return update_num / N;
}

void initialize()
{
	evals = 0;
    pop_tasks.resize(ntasks, Population());
    for (int i = 0; i < ntasks; i++)
    {
        EA_solver->InitializePopulation(pop_tasks[i], task_evals[i]);
        evals += pop_tasks[i].size();
        record_tasks[i].Initialize(island_infos[i], problem_infos[i], ea_infos[i]);
        task_sels[i].Initialize(island_infos[i]);
        fprintf(stderr, "task %d init best fitness %.4f\n", i+1, EA_solver->FindBestIndividual(pop_tasks[i]).fitness_value);
    }
    tsf_probs.resize(ntasks, (p_tsf_ub + p_tsf_lb) / 2);
    rewards_self.resize(ntasks, 0.0);
    rewards_other.resize(ntasks, 0.0);
    tsf_cnt.resize(ntasks, 0);
}

void uninitialize()
{
    pop_tasks.clear();
    tsf_probs.clear();
    rewards_self.clear();
    rewards_other.clear();
    tsf_cnt.clear();
    for(int i = 0; i < ntasks; i++)
    {
        record_tasks[i].Uninitialize();
        task_sels[i].UnInitialize();
    }
}

void AEMTO()
{
	initialize();
	int g = 0;
	while (g < MAX_GENS)
    {
        for (int task_id = 0; task_id < ntasks; task_id++)
        {
            if (MTO && (rand_.RandRealUnif(0.0, 1.0) < tsf_probs[task_id]))
            {
                unordered_map<int, int> success_table;
                Population other_pop = select_from_others(task_id); 
                real r_tsf = Reuse(task_id, pop_tasks[task_id], other_pop, success_table);
                rewards_other[task_id] = rewards_other[task_id] * alpha + (1 - alpha) * r_tsf;
                task_sels[task_id].update_import_pdf(success_table, g);
                tsf_cnt[task_id]++;
            } else {
                real r_self = EA_solver->Run(pop_tasks[task_id], task_evals[task_id]);
                rewards_self[task_id] = rewards_self[task_id] * alpha + (1 - alpha) * r_self;
            }
            tsf_probs[task_id] = p_tsf_lb 
                    + rewards_other[task_id] * (p_tsf_ub - p_tsf_lb) 
                    / (rewards_other[task_id] + rewards_self[task_id] + eps);
        } 
		if ((g + 1) % record_tasks[0].RECORD_INTERVAL == 0 || 
            (g == 0))
        {
			for (int i = 0; i < ntasks; i++)
            {
                Individual ind = EA_solver->FindBestIndividual(pop_tasks[i]);
                real bestf = ind.fitness_value; 
                fprintf(stderr, "task %d; runs %d/%d; gens %d/%d; tsf count %d; "
                                "tsf_prob %.4f; bestf %.12f\n",
                       i + 1, job+1, RUNS, g+1, MAX_GENS, tsf_cnt[i], tsf_probs[i], bestf);
                RecordInfo info;
                info.best_fitness = bestf;
                info.generation = g+1;
                info.comm_time = 0;
                info.time = 0;
                record_tasks[i].RecordInfos(info);
            }
        }
        g++;
	}
    for (int i = 0; i < ntasks; i++)
    {
        Individual ind = EA_solver->FindBestIndividual(pop_tasks[i]);
        real bestf = ind.fitness_value;
        stringstream ss;
        for (auto e : ind.elements)
        {
            ss << e << ", ";
        }
        fprintf(stderr, "task %d run_id %d, final results: [%s %.12f] \n", 
                i+1, job+1, ss.str().c_str(), bestf);
        record_tasks[i].FlushInfos();
    }
    uninitialize();
}
int main(int argc, char* argv[])
{
	clock_t start = clock();
    global_init(argc, argv);
	for (job = 0; job < RUNS; job++)
	{
        double time_start = get_wall_time();
		srand((job+1)*10000);
		AEMTO();
        cout << "one run cost time " << get_wall_time() - time_start << endl;
	}
    global_deinit();
	cout << "total time cost: " << (double)(clock() - start) / CLOCKS_PER_SEC 
          << " seconds for " << RUNS << " RUNS" << endl;
	return 0;
}
