#include <sstream>
#include <algorithm>
#include "config.h"
#include "util.h"
#include "record.h"
#include "random.h"
#include "EA.h"
#include "set_parameters.h"

using namespace std;

const Real eps = 1e-10;
int run_id = 0;
int ntasks;
Real p_tsf_lb = 0.05; // lower bound of transfer probability
Real p_tsf_ub = 0.70; // upper ...
Real alpha=0.3; // reward update rate
Real pbase=0.3;

vector<Real> tasks_tsf_probs; // transfer prob of each task
vector<Real> tasks_rewards_self; // rewards of self eval of each task
vector<Real> tasks_rewards_other; // rewards of inter-task transfer of each task
vector<vector<Real>> tasks_selection_pdf; // selection pdf of each task
vector<vector<Real>> tasks_selection_rewards; // selection rewards of each task
vector<int> tasks_tsf_cnt; //record transfer count of each task
vector<unique_ptr<Evaluator> > task_evals; // evaluator of each task
vector<ProblemInfo> problem_infos;
unique_ptr<EA> EA_solver;
EAInfo EA_info;

vector<Population> pop_tasks; // population of each task
vector<Record> record_tasks; // record of each task
Random rand_;
Args args;


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
    ntasks = (int)args.total_tasks.size();
    fprintf(stderr, "=================== INFO ============\n");
    fprintf(stderr, "total tasks = %d; total runs = %d\n", ntasks, args.total_runs);
    fprintf(stderr, "pop_size x m = %dx%d\n", args.popsize, ntasks);
    fprintf(stderr, "Gmax = %d\n", args.Gmax);
    fprintf(stderr, "results dir = %s\n", args.results_dir.c_str());
    fprintf(stderr, "EA solver = %s.\n", EA_info.STO.c_str());
    fprintf(stderr, "problem set = %s.\n", args.problem_set.c_str());
    fprintf(stderr, "MTO = %d\n", args.MTO);

    return 0;
}

Population select_from_others(const unordered_map<int, int> &select_table)
{
    Population selected_individuals;
    for (auto ele : select_table)
    {
        int other_task_id = ele.first;
        int num_sel = ele.second;
        vector<Real> fitnesses;
        const Population &pop_other = pop_tasks[other_task_id];
        for(const auto & e : pop_other) {
            fitnesses.push_back(e.fitness_value);
        }
        vector<int> indices = rand_.roulette_sample(fitnesses, num_sel);
        for (const auto &i : indices)
        {
            Individual ind = pop_other.at(i);
            ind.skill_factor = other_task_id;
            selected_individuals.emplace_back(ind);
        }
    }
    return selected_individuals;
}

Real Reuse(int task_id, Population &pop, Population &other_pop, 
        unordered_map<int, int> &success_insert_table)
{
    int other_pop_size = other_pop.size();
    if(other_pop_size == 0) return 0.0;
    vector<int> rand_indexs = rand_.Permutate(other_pop_size, other_pop_size);
    int N = pop.size();
    Real update_num = 0;
    for (int k = 0; k < N; k++)
    {
        Individual mu = other_pop.at(rand_indexs[k % other_pop_size]);
        Individual x = pop.at(k);
        Real cr = rand_.RandRealUnif(EA_info.LKTCR, EA_info.UKTCR);
        Individual c = binomial_crossover(x, mu, cr);
        c.fitness_value = task_evals[task_id]->EvaluateFitness(c.elements);
        if (c.fitness_value < x.fitness_value)
        {
            pop[k] = c;
            pop[k].skill_factor = -1;
            success_insert_table[mu.skill_factor]++;
            update_num+=1;
        }
    }
    return update_num / N;
}

void update_selection_pdf(int task_id, 
                          unordered_map<int, int> &success_table, 
                          unordered_map<int, int> &select_table) {
    auto &task_rewards = tasks_selection_rewards[task_id];
    for (const auto &e: select_table) {
        int other_task_id = e.first;
        int n = e.second; // number of success preserved
        int ns = 0;
        if (success_table.find(other_task_id) != success_table.end()) {
            ns = success_table[other_task_id];
        }
        task_rewards[other_task_id] = (1 - alpha) * ((Real)ns / (Real)n) + 
                        task_rewards[other_task_id] * alpha;
    }
    Real rewards_sum = accumulate(task_rewards.begin(), task_rewards.end(), 0.0);
    Real pmin = pbase / (Real)(ntasks - 1);
    for (int j = 0; j < tasks_selection_pdf[task_id].size(); j++) {
        if (j == task_id) continue; //skip update the 0-probabilitly of selecting task itself
        tasks_selection_pdf[task_id][j] = pmin + 
            (1 - (ntasks-1)*pmin) * (task_rewards[j] / (rewards_sum + eps)); 
    }
}


void initialize()
{
    pop_tasks.resize(ntasks, Population());
    for (int i = 0; i < ntasks; i++)
    {
        pop_tasks[i].resize(args.popsize, Individual(args.UDim));
        EA_solver->InitializePopulation(pop_tasks[i], task_evals[i]);
        tasks_selection_pdf.push_back(vector<Real>(ntasks, 1.0 / (ntasks - 1)));
        tasks_selection_pdf.back()[i] = 0.0; // prob of selecting task itself is 0.0
        tasks_selection_rewards.push_back(vector<Real>(ntasks, 0.0));
        fprintf(stderr, "task %d init best fitness %.4f\n", 
                        i+1, EA_solver->FindBestIndividual(pop_tasks[i]).fitness_value);
    }
    tasks_tsf_probs.resize(ntasks, (p_tsf_ub + p_tsf_lb) / 2);
    tasks_rewards_self.resize(ntasks, 0.0);
    tasks_rewards_other.resize(ntasks, 0.0);
    tasks_tsf_cnt.resize(ntasks, 0);
}

void uninitialize()
{
    pop_tasks.clear();
    tasks_tsf_probs.clear();
    tasks_rewards_self.clear();
    tasks_rewards_other.clear();
    tasks_tsf_cnt.clear();
    tasks_selection_pdf.clear();
    tasks_selection_rewards.clear();
}

void AEMTO()
{
	initialize();
	int g = 0;
	while (g < args.Gmax)
    {
        for (int task_id = 0; task_id < ntasks; task_id++)
        {
            if (args.MTO && (rand_.RandRealUnif(0.0, 1.0) <= tasks_tsf_probs[task_id]))
            {
                auto select_table = rand_.sus_sampleV2(
                    tasks_selection_pdf[task_id], args.popsize);
                auto other_pop = select_from_others(select_table); 
                unordered_map<int, int> success_table;
                Real r_tsf = Reuse(task_id, pop_tasks[task_id], other_pop, success_table);
                tasks_rewards_other[task_id] = tasks_rewards_other[task_id] * alpha + (1 - alpha) * r_tsf;
                update_selection_pdf(task_id, success_table, select_table);
                tasks_tsf_cnt[task_id]++;
            } else {
                Real r_self = EA_solver->Run(pop_tasks[task_id], task_evals[task_id]);
                tasks_rewards_self[task_id] = tasks_rewards_self[task_id] * alpha + (1 - alpha) * r_self;
            }
            tasks_tsf_probs[task_id] = p_tsf_lb 
                    + tasks_rewards_other[task_id] * (p_tsf_ub - p_tsf_lb) 
                    / (tasks_rewards_other[task_id] + tasks_rewards_self[task_id] + eps);
        } 
		if ((g + 1) % args.record_interval == 0 || 
            (g == 0))
        {
			for (int i = 0; i < ntasks; i++)
            {
                auto ind = EA_solver->FindBestIndividual(pop_tasks[i]);
                Real bestf = ind.fitness_value; 
                fprintf(stderr, "task %d; runs %d/%d; gens %d/%d; tsf count %d; "
                                "tsf_prob %.4f; bestf %.12f\n",
                                 i + 1, run_id + 1, args.total_runs, g+1, args.Gmax, tasks_tsf_cnt[i], 
                                 tasks_tsf_probs[i], bestf);
                RecordInfo info;
                info.best_fitness = bestf;
                info.generation = g+1;
                record_tasks[i].RecordInfos(info);
            }
        }
        g++;
	}
    for (int i = 0; i < ntasks; i++)
    {
        auto ind = EA_solver->FindBestIndividual(pop_tasks[i]);
        Real bestf = ind.fitness_value;
        stringstream ss;
        for (auto e : ind.elements) {
            ss << e << ", ";
        }
        fprintf(stderr, "task %d run_id %d, final solution (in unified space): [%s]; "
                        "final bestf: %.12f \n", 
                        i+1, run_id + 1, ss.str().c_str(), 
                        bestf);
        record_tasks[i].FlushInfos(run_id);
        record_tasks[i].Clear();
    }
    uninitialize();
}
int main(int argc, char* argv[])
{
	auto start = get_wall_time();
    global_init(argc, argv);
	for (run_id = 0; run_id < args.total_runs; run_id++)
	{
        double time_start = get_wall_time();
		srand((run_id + 1)*10000);
		AEMTO();
        cout << "one run cost time " << get_wall_time() - time_start << " seconds" << endl;
	}
	cout << "total time cost: " << get_wall_time() - start
          << " seconds for " << args.total_runs << " runs" << endl;
	return 0;
}
