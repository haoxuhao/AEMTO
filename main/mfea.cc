#include <omp.h>
#include "config.h"
#include "set_parameters.h"
#include "evaluator.h"
#include "EA.h"
#include "random.h"
#include "util.h"
#include "record.h"
#include "brent.hpp"
#include <mutex>

using namespace brent;
using namespace std;

struct Chromosome
{
    vector<real> elements;
    vector<real> factorial_costs;
    vector<int> factorial_ranks;
    real scalar_fitness;
    int skill_factor;
};
typedef Chromosome Chro;

const int RECORD_FRE = 100;
const real pm = 1;
bool MTO = true;
int generation;	
int	evals;	
int job;
int ntasks;
int pop_size;
int D;
int RUNS = 1;
int MAX_GENS = 1000;
long MAX_EVALS = 100000;

vector<int> vars;
vector<vector<real>> RMP;
vector<Chromosome> pop_all;
vector<pair<real, int>> best_objs; //best <obj, index> of each task.

vector<unique_ptr<Evaluator>> manytask_funs;
mutex mutexes[50];
vector<IslandInfo> island_infos;
vector<ProblemInfo> problem_infos;
vector<EAInfo> ea_infos;
vector<Record> record_tasks;
Random random_;
string result_dir;
GA_CPU *EA_solver;

vector<int> find_in_pop(const vector<Chro> &pop, int skill_factor);

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

        vars.push_back(problem_info.calc_dim);
    }

    MAX_EVALS = island_info.run_param.FEs;
    MAX_GENS = MAX_EVALS / island_info.island_size;
    ntasks = (int)args.total_tasks.size();

    if (EA_info.STO == "GA")
	{
		EA_solver = new GA_CPU();
        EA_solver->Initialize(island_info, problem_info, EA_info);
        
	} else{
        fprintf(stderr, "Error no EA solver found for MFEA: %s.\n", EA_info.STO.c_str());
        exit(-1);
    }
    fprintf(stderr, "=================== Init INFO ============\n");
    fprintf(stderr, "total tasks = %d; total runs = %d\n", ntasks, RUNS);
    fprintf(stderr, "pop_sizexm = %dx%d\n", island_info.island_size, ntasks);
    fprintf(stderr, "MAX_EVALS = %ldx%d; MAX_GENS = %d\n", MAX_EVALS, ntasks, MAX_GENS);
    fprintf(stderr, "results dir = %s\n", result_dir.c_str());
    fprintf(stderr, "EA solver = %s.\n", EA_info.STO.c_str());
    fprintf(stderr, "MTO = %d\n", MTO);
    
    return 0;
}

int global_deinit()
{
    for(int i = 0; i < manytask_funs.size(); i++)
    {
        manytask_funs[i]->Uninitialize();
    }
	EA_solver->Uninitialize();
    delete EA_solver;
    return 0;
}

void init_RMP(real init_rmp = 0.3)
{
    for(int i = 0; i < ntasks; i++)
    {
        RMP.push_back(vector<real>(ntasks, 0.0));
        for(int j = 0; j < ntasks; j++)
        {
            if (i == j)
            {
                RMP[i][j] = 1.0;
            }else{
                RMP[i][j] = init_rmp;
            }
        }
    }
}

/********* Update rmp matrix related code begin *******/
struct ProbMat
{
    vector<vector<real>> probmat; // m * n
};

class MyFunc : public func_base
{
private:
    ProbMat *popdata;
public:
    virtual double operator() (double rmp)
    {
        real f = 0;
        int nsamples = popdata[0].probmat.size(); 
        real p_self, p_other;
        real other_ratio = 0.5 * (ntasks - 1)*rmp / ntasks;

        for (int k = 0; k < nsamples; k++)
        {
            p_self = popdata[0].probmat[k][0]*(1 - other_ratio);
            p_other = popdata[0].probmat[k][1]*(other_ratio);
            f += -log(p_self + p_other);
        }
        nsamples = popdata[1].probmat.size(); 
        for (int k = 0; k < nsamples; k++)
        {
            p_self = popdata[1].probmat[k][1]*(1 - other_ratio);
            p_other = popdata[1].probmat[k][0]*(other_ratio);
            f += -log(p_self + p_other);
        }
        return f; 
    };

    MyFunc(ProbMat *popdata)
    {
        this->popdata = popdata;
    }
    ~MyFunc()
    {
    }
};

/**
 * return mean, std, n of subpop i
*/
tuple<vector<real>, vector<real>, int, vector<int>> probmodel(vector<Chro> &pop, int task)
{
    vector<int> task_indices = find_in_pop(pop, task);
    vector<real> mean(D, 0);
    vector<real> std(D, 0);
    int nsamples = task_indices.size();
    int nrandsamples = 0.1 * nsamples;
    vector<vector<real>> rand_mat = random_.RandReal2D(nrandsamples, D);

    // #pragma omp parallel for
    for (int k = 0; k < D; k++)
    {
        real s = 0;
        for(int i = 0; i < task_indices.size(); i++)
        {
            s += pop.at(task_indices[i]).elements[k];
        }
        for(int i = 0; i < nrandsamples; i++)
        {
            s += rand_mat.at(i).at(k);
        }
        mean[k] = s / (nsamples + nrandsamples);
        real v = 0;
        for(int i = 0; i < task_indices.size(); i++)
        {
            v += pow((pop.at(task_indices[i]).elements[k] - mean[k]), 2);
        }
        for(int i = 0; i < nrandsamples; i++)
        {
            v += pow(rand_mat.at(i).at(k) - mean[k], 2);
        }
        std[k] = sqrt(v / (nsamples + nrandsamples));
    }
    return make_tuple(mean, std, nsamples, task_indices);
}

real norm_pdf(real x, real m, real s)
{
    static const real inv_sqrt_2pi = 0.3989422804014327;
    float a = (x - m) / s;
    real p = inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
    return p;
}

int update_RMP(vector<Chro> &pop)
{
    vector<tuple<vector<real>, vector<real>, int, vector<int>>> probmodels; //mean std nsamples, indices
    // O(m*D*N)
    real s = get_wall_time();
    for (int i = 0; i < ntasks; i++)
    {
        probmodels.emplace_back(probmodel(pop, i));
    }
    printf("prob model build time %f\n", get_wall_time() - s);

    // printf("init probmodel ok.\n");
    vector<pair<int, int>> items_to_calc;
    vector<ProbMat*> probmats;
    for (int i = 0; i < ntasks; i++)
    {
        for(int j = i+1; j < ntasks; j++)
        {
            items_to_calc.emplace_back(make_pair(i, j));
            ProbMat *mats = new ProbMat[2];
            probmats.push_back(mats);
        }
    }
    int num_items = items_to_calc.size();
    // printf("prepare parallel rmp calc ok, number to calc %d.\n", num_items);
    real t = r8_epsilon();
    real e = t;
    real a = 0;
    real b = 1;
    real m = 0;
    real c = (a + b) / 2;
    // O(m*m*N*D)
    real s2 = get_wall_time();
    // #pragma omp parallel for 
    for(int n = 0; n < num_items; n++)
    {
        int i = items_to_calc.at(n).first;
        int j = items_to_calc.at(n).second;
        int dims = min(vars.at(i), vars.at(j));

        const vector<real> &mean_i = std::get<0>(probmodels[i]);
        const vector<real> &mean_j = std::get<0>(probmodels[j]);

        const vector<real> &std_i = std::get<1>(probmodels[i]);
        const vector<real> &std_j = std::get<1>(probmodels[j]);

        int nsamples_i = std::get<2>(probmodels[i]);
        int nsamples_j = std::get<2>(probmodels[j]);

        const vector<int> &indices_i = std::get<3>(probmodels[i]);
        const vector<int> &indices_j = std::get<3>(probmodels[j]);

        probmats[n][0].probmat.resize(nsamples_i, vector<real>(2, 1.0));
        probmats[n][1].probmat.resize(nsamples_j, vector<real>(2, 1.0));
        // real s1 = get_wall_time();
        for (int k = 0; k < nsamples_i; k++)
        {
            for (int l = 0; l < dims; l++)
            {
                probmats[n][0].probmat[k][0] *= norm_pdf(pop_all.at(indices_i.at(k)).elements[l], mean_i.at(l), std_i.at(l));
                probmats[n][0].probmat[k][1] *= norm_pdf(pop_all.at(indices_i.at(k)).elements[l], mean_j.at(l), std_j.at(l));
            }
        }
        for (int k = 0; k < nsamples_j; k++)
        {
            for (int l = 0; l < dims; l++)
            {
                probmats[n][1].probmat[k][0] *= norm_pdf(pop_all.at(indices_j.at(k)).elements[l], mean_i.at(l), std_i.at(l));
                probmats[n][1].probmat[k][1] *= norm_pdf(pop_all.at(indices_j.at(k)).elements[l], mean_j.at(l), std_j.at(l));
            }
        }
        // printf("build mixture model 1 cost time %f\n", get_wall_time() - s1);
        real rmp = 0.0; //init guess
        MyFunc f3 = MyFunc(probmats[n]);
        // real ret = local_min(a, b, t, f3, rmp);
        real start = get_wall_time();
        real ret = glomin(a, b, c, m, e, t, f3, rmp);
        // printf("fminbnd cost time %f s\n", get_wall_time() - start);
        RMP[i][j] = max(0.0, rmp + random_.RandRealNormal(0, 0.01));
        RMP[i][j] = min(RMP[i][j], 1.0);
        RMP[j][i] = RMP[i][j];
        // printf("optimal rmp found %.4f; RMP (%d, %d): %.3f\n", rmp, i, j, RMP[i][j]);
        // exit(2);
    }
    printf("calculate rmp matrix cost time %f\n", get_wall_time() - s2);
    //destroy the probmats
    for(int n = 0; n < num_items; n++)
    {
        delete []probmats[n];
    }
    return 0;
}



/********* Update rmp matrix related code end **********/


int argmax(vector<real> x, int skip_index)
{
    int max_id = -1;
    real max_val = -1;
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
vector<int> find_in_pop(const vector<Chro> &pop, int skill_factor)
{
    vector<int> find_indices;
    for (int i = 0; i < pop.size(); i++)
    {
        if (pop[i].skill_factor == skill_factor)
        {
            find_indices.push_back(i);
        }
    }
    return find_indices;
}

pair<int, real> best_in_pop(const vector<Chro> &p, int skill)
{
    real bestf = DBL_MAX;
    int idx = -1;
    for(int i = 0; i < p.size(); i++)
    {
        if(p[i].skill_factor == skill && p[i].factorial_costs[skill] < bestf)
        {
            idx = i;
            bestf = p[i].factorial_costs[skill];
        }
    }
    return make_pair(idx, bestf);
}

void initialized()
{
	evals = 0;
    int sub_pop_size = island_infos.at(0).island_size;
    if (sub_pop_size % 2 != 0) sub_pop_size++; // subpopsize must be even
    D = problem_infos.at(0).dim;
    pop_size = sub_pop_size * ntasks;
    pop_all.clear();
    fprintf(stderr, "population size %d, max dim %d\n", pop_size, D);
    for (int i = 0; i < ntasks; i++)
    {
        // #pragma omp parallel for 
        for (int j = 0; j < sub_pop_size; j++)
        {
            Chromosome c;
            for (int k = 0; k < D; k++)
            {
                c.elements.push_back(random_.RandRealUnif(0, 1));
            }
            c.skill_factor = i;
            c.factorial_costs = vector<real>(ntasks, DBL_MAX);
            c.factorial_ranks = vector<int>(ntasks, 2*pop_size + 1);
            c.factorial_costs.at(i) = manytask_funs.at(i)->EvaluateFitness(c.elements);
            c.factorial_ranks.at(i) = j+1;
            // mutexes[0].lock();
            pop_all.emplace_back(c);
            evals++;
            // mutexes[0].unlock();
        }
        // the last individual be the initial best individual.
        best_objs.push_back(std::make_pair(pop_all.back().factorial_costs[i], pop_all.size() - 1));
        record_tasks[i].Initialize(island_infos[i], problem_infos[i], ea_infos[i]);
    }
    if(MTO)
    {
        init_RMP(0.3);
    }
    else
    {
        init_RMP(0);
    }
    cerr << "initialized. " << endl;
}

void uninitialized()
{
    pop_all.clear();
    for(int i = 0; i < ntasks; i++)
    {
        record_tasks[i].Uninitialize();
    }
}

vector<Chro> variation()
{
    vector<int> inorder = random_.Permutate(pop_size, pop_size);
    vector<Chromosome> offspring;
    int idx_p1, idx_p2;
    int half_size = pop_size / 2;
    for (int i = 0; i < half_size; i++)
    {
        idx_p1 = inorder.at(i); 
        idx_p2 = inorder.at(i + half_size - 1);
        Chromosome p1 = pop_all.at(idx_p1);
        Chromosome p2 = pop_all.at(idx_p2);
        Chromosome c1, c2;
        real rmp = RMP[p1.skill_factor][p2.skill_factor];
        vector<real> cf = EA_solver->generate_cf(D);
        if (p1.skill_factor == p2.skill_factor)
        {
            //(SBX + uniform) + mutation
            c1.elements = EA_solver->crossover(p1.elements, p2.elements, cf);
            c2.elements = EA_solver->crossover(p2.elements, p1.elements, cf);
            if (random_.RandRealUnif(0, 1.0) < pm)
            {
                EA_solver->mutate(c1.elements);
                EA_solver->mutate(c2.elements);
            }
            EA_solver->swap(c1.elements, c2.elements);
            c1.skill_factor = p1.skill_factor;
            c2.skill_factor = p2.skill_factor;
        }
        else if (random_.RandRealUnif(0, 1.0) < rmp)
        {
            c1.elements = EA_solver->crossover(p1.elements, p2.elements, cf);
            c2.elements = EA_solver->crossover(p2.elements, p1.elements, cf);
            if (random_.RandRealUnif(0, 1) < pm)
            {
                EA_solver->mutate(c1.elements);
                EA_solver->mutate(c2.elements);
            }
            if (random_.RandRealUnif(0, 1) < 0.5)
            {
                c1.skill_factor = p1.skill_factor;
            }else
            {
                c1.skill_factor = p2.skill_factor;
            }
            if (random_.RandRealUnif(0, 1) < 0.5)
            {
                c2.skill_factor = p2.skill_factor;
            }else
            {
                c2.skill_factor = p1.skill_factor;
            }
        }else
        {
            // printf("different skill factors p1 %d, p2 %d\n", p1.skill_factor, p2.skill_factor);
            /* select another unique individual the same as p1 */
            vector<int> sol1 = find_in_pop(pop_all, p1.skill_factor);
            int num_sol1 = sol1.size();
            int sel1 = sol1[random_.Permutate(num_sol1, 1)[0]];
            while(sel1 == idx_p1)
            {
                sel1 = sol1[random_.Permutate(num_sol1, 1)[0]];
            }
            // printf("sub pop %d, size %d, p1 index %d, sel1 index %d\n", p1.skill_factor, num_sol1, inorder[idx_p1], sel1);
            Chro tmpc1;
            c1.elements = EA_solver->crossover(p1.elements, pop_all.at(sel1).elements, cf);
            tmpc1.elements = EA_solver->crossover(pop_all.at(sel1).elements, p1.elements, cf);
            if (random_.RandRealUnif(0, 1) < pm) 
            {
                EA_solver->mutate(c1.elements);
                EA_solver->mutate(tmpc1.elements);
            }
            EA_solver->swap(c1.elements, tmpc1.elements);
            c1.skill_factor = p1.skill_factor;

            /* select another unique individual the same as p2 */
            vector<int> sol2 = find_in_pop(pop_all, p2.skill_factor);
            int num_sol2 = sol2.size();
            int sel2 = sol2[random_.Permutate(num_sol2, 1)[0]];
            while(sel2 == idx_p2)
            {
                sel2 = sol2[random_.Permutate(num_sol2, 1)[0]]; 
            }
            // printf("sub pop %d, size %d, p2 index %d, sel2 index %d\n", p2.skill_factor, num_sol2, inorder[idx_p2], sel2);
            Chro tmpc2;
            c2.elements = EA_solver->crossover(p2.elements, pop_all.at(sel2).elements, cf);
            tmpc2.elements = EA_solver->crossover(pop_all.at(sel2).elements, p2.elements, cf);
            if (random_.RandRealUnif(0, 1) < pm) 
            {
                EA_solver->mutate(c2.elements);
                EA_solver->mutate(tmpc2.elements);
            }
            EA_solver->swap(c2.elements, tmpc2.elements);
            c2.skill_factor = p2.skill_factor;
        }
        offspring.emplace_back(c1);
        offspring.emplace_back(c2);
    }
    return offspring;
}

void reproduce()
{
    clock_t start = clock();
    vector<Chro> offspring = variation();
    // fprintf(stderr, "offspring cost time %.3f s\n", (real)(clock() - start) / CLOCKS_PER_SEC); 

    start = clock();

    // #pragma omp parallel
    {   
        // #pragma omp for
        for (int i = 0; i < offspring.size(); i++)
        {
            Chro &c = offspring.at(i);
            c.factorial_costs = vector<real>(ntasks, DBL_MAX);
            c.factorial_ranks = vector<int>(ntasks, 2 * pop_size + 1);
            c.factorial_costs[c.skill_factor] = \
                manytask_funs[c.skill_factor]->EvaluateFitness(c.elements); //, mutexes[c.skill_factor]
        }
    }
    evals += offspring.size();
    // fprintf(stderr, "eval offspring cost time %.3f s\n", (real)(clock() - start) / CLOCKS_PER_SEC); 
 
    start = clock();
    vector<Chro> inter_pop;
    inter_pop.insert(inter_pop.end(), pop_all.begin(), pop_all.end());
    inter_pop.insert(inter_pop.end(), offspring.begin(), offspring.end());
    int inter_pop_size = inter_pop.size();
    vector<real> tmp_vec(inter_pop_size, 0.0);
    for (int i = 0; i < ntasks; i++)
    {
        for(int j = 0; j < inter_pop.size(); j++)
        {
            tmp_vec[j] = inter_pop[j].factorial_costs[i];
        }
        vector<int> sorted_indices = argsort(tmp_vec);
        for(int j = 0; j < sorted_indices.size(); j++)
        {
            inter_pop[sorted_indices[j]].factorial_ranks[i] = j + 1;
        }
    }
    for (int i = 0; i < inter_pop_size; i++)
    {
        vector<int> &ranks = inter_pop[i].factorial_ranks;
        int min_idx = inter_pop[i].skill_factor;
        int min_rank = ranks[min_idx];
        if (MTO){
            for(int k = 0; k < ranks.size(); k++)
            {
                if(ranks[k] < min_rank)
                {
                    min_idx = k;
                    min_rank = ranks[k];
                }
            }
        }
        if(min_idx != inter_pop[i].skill_factor && !MTO)
        {
            vector<int> sub_pop_0 = find_in_pop(inter_pop, 0);
            vector<int> sub_pop_1 = find_in_pop(inter_pop, 1);
            fprintf(stderr, "inter_pop_index %d skill factor shift, transfer of skill factor %d -> %d\n",
                 i, inter_pop[i].skill_factor, min_idx);
            stringstream ss;
            for(const auto &e : inter_pop[i].factorial_costs)
            {
                ss << e << ", ";
            }
            fprintf(stderr, "factorial costs %s\n", ss.str().c_str());
            ss.str("");
            for(const auto &e : inter_pop[i].factorial_ranks)
            {
                ss << e << ", ";
            }
            fprintf(stderr, "factorial ranks %s\n", ss.str().c_str());
        }
        inter_pop[i].skill_factor = min_idx;
        inter_pop[i].scalar_fitness = 1.0 / ranks[min_idx];
        tmp_vec[i] = inter_pop[i].scalar_fitness;
    }
    // fprintf(stderr, "scalacr fitness cost time %.3f s\n", (real)(clock() - start) / CLOCKS_PER_SEC); 

    clock_t sort_start = clock();
    vector<int> sorted_pop_indices = argsort(tmp_vec);
    // fprintf(stderr, "sort cost time %.3f s\n", (real)(clock() - sort_start) / CLOCKS_PER_SEC);
    for(int i = 0; i < pop_size; i++)
    {
        int sort_id = sorted_pop_indices[inter_pop_size - i - 1];
        Chro new_c = inter_pop.at(sort_id);
        pop_all[i] = new_c;
        if (new_c.scalar_fitness == 1)
        {
            best_objs[new_c.skill_factor].first = \
                new_c.factorial_costs.at(new_c.skill_factor);
            best_objs[new_c.skill_factor].second = i;
        }
    }
}

void MFEA()
{
	initialized();
	generation = 0;
    real bestf = 0;

	while (generation < MAX_GENS)
    {
        double start = get_wall_time();
	    if (MTO)
	    {
        	// 1
        	// auto t_start = std::chrono::high_resolution_clock::now();
       	 	update_RMP(pop_all);
        	// auto t_end = std::chrono::high_resolution_clock::now();
        	// printf("update rmp cost time %.3f ms\n", std::chrono::duration<double, std::milli>(t_end-t_start).count());
	    }
        // 2
        reproduce();

		//Print the current best
		if ((generation + 1) % record_tasks[0].RECORD_INTERVAL == 0
             || (generation == 0))
        {
			for (int i = 0; i < ntasks; i++)
            {
                Chromosome ind = pop_all.at(best_objs.at(i).second);
                bestf = best_objs.at(i).first;
                fprintf(stderr, "island %d; runs %d/%d; gens %d/%d; bestf %.12f\n", i, job+1, RUNS, generation+1, MAX_GENS, bestf);
                //  stringstream ss;
                //  for(int k = 0; k < ntasks; k++)
                //  {
                //      if(k != ntasks - 1)
                //      {
                //          ss << RMP[i][k] << ", ";
                //      }else
                //      {
                //          ss << RMP[i][k];
                //      }
                // }
                // fprintf(stdout, "island %d RMPs; runs %d/%d; gens %d/%d: [%s].\n",
                //      i, job+1, RUNS, generation+1, MAX_GENS, ss.str().c_str());

                RecordInfo info;
                info.best_fitness = bestf;
                info.generation = generation+1;
                // info.elements = ind.elements;
                info.comm_time = 0;
                info.time = 0;
                record_tasks[i].RecordInfos(info);
            }
        }
        generation++;
        cerr << "one generation cost time " << get_wall_time() - start << endl;
	}
    for (int i = 0; i < ntasks; i++)
    {
        auto best = best_in_pop(pop_all, i);
        Chromosome &ind = pop_all.at(best.first);
        bestf = best.second;
        stringstream ss;
        for (auto e : ind.elements)
        {
            ss << e << ", ";
        }
        fprintf(stderr, "island %d run_id %d, final results: [%s %.12f] \n", i, job+1, ss.str().c_str(), bestf);
        record_tasks[i].FlushInfos();
    }
    
    uninitialized();

}
int main(int argc, char* argv[])
{
    auto t_start = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(8);
    global_init(argc, argv);
	for (job = 7; job < RUNS; job++)
	{
		srand((job+1)*10000);
		MFEA();
        auto t_end = std::chrono::high_resolution_clock::now(); 
        cerr << (job + 1) << " run cost time " << std::chrono::duration<double, std::milli>(t_end-t_start).count() / 1000.0 << " s\n";
	}
    global_deinit();
    auto t_end = std::chrono::high_resolution_clock::now();
	cerr << "total wall clock time: " << std::chrono::duration<double, std::milli>(t_end-t_start).count() / 1000.0 << " s for " << RUNS << " RUNS" << endl;
	return 0;
}
