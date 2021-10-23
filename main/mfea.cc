#include "config.h"
#include "set_parameters.h"
#include "evaluator.h"
#include "EA.h"
#include "random.h"
#include "util.h"
#include "record.h"
#include "brent.hpp"

using namespace brent;
using namespace std;

struct Chromosome
{
    vector<Real> elements;
    vector<Real> factorial_costs;
    vector<int> factorial_ranks;
    Real scalar_fitness;
    int skill_factor;
};
typedef Chromosome Chro;


const Real pm = 1; // probability of mutation
bool MFEA2 = true; // whether update the RMP matrix adaptively
int	evals;	
int run_id;
int ntasks; // number of tasks
int pop_size; // pop size of whole population
int D; // Dim of unified searched space

vector<int> calc_dims; // evaluation dim of each task
vector<vector<Real>> RMP; // random mating matrix
vector<Chromosome> pop_all; // one population for all tasks
vector<pair<Real, int>> best_objs; //record the best individual of each task <obj, index in the pop> 
vector<unique_ptr<Evaluator>> task_evals; // evaluator of each task
vector<ProblemInfo> problem_infos; // problem info of each task
vector<Record> record_tasks; // record class of each task
Random random_; // Random helper class
unique_ptr<GA> EA_solver; // GA solver 
EAInfo EA_info; // params of GA
Args args; // global args

vector<int> find_in_pop(const vector<Chro> &pop, int skill_factor);

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
        calc_dims.push_back(problem_infos[i].calc_dim);
    }

    ntasks = (int)args.total_tasks.size();
    EA_info.STO = "GA";
    if (EA_info.STO == "GA")
	{
		EA_solver.reset(new GA(problem_infos[0], EA_info));
	} else{
        fprintf(stderr, "Error no EA solver found for MFEA: %s.\n", EA_info.STO.c_str());
        exit(-1);
    }
    fprintf(stderr, "=================== Init INFO ============\n");
    fprintf(stderr, "total tasks = %d; total runs = %d\n", ntasks, args.total_runs);
    fprintf(stderr, "pop_sizexm = %dx%d\n", args.popsize, ntasks);
    fprintf(stderr, "args.Gmax = %d\n", args.Gmax);
    fprintf(stderr, "results dir = %s\n", args.results_dir.c_str());
    fprintf(stderr, "EA solver = %s.\n", EA_info.STO.c_str());
    fprintf(stderr, "MTO = %d\n", args.MTO);
    
    return 0;
}

void init_RMP(Real init_rmp = 0.3)
{
    for(int i = 0; i < ntasks; i++)
    {
        RMP.push_back(vector<Real>(ntasks, 0.0));
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
    vector<vector<Real>> probmat; // m * n
};

class MyFunc : public func_base
{
private:
    ProbMat *popdata;
public:
    virtual double operator() (double rmp)
    {
        Real f = 0;
        int nsamples = popdata[0].probmat.size(); 
        Real p_self, p_other;
        Real other_ratio = 0.5 * (ntasks - 1)*rmp / ntasks;

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
tuple<vector<Real>, vector<Real>, int, vector<int>> probmodel(vector<Chro> &pop, int task)
{
    vector<int> task_indices = find_in_pop(pop, task);
    vector<Real> mean(D, 0);
    vector<Real> std(D, 0);
    int nsamples = task_indices.size();
    int nrandsamples = 0.1 * nsamples;
    vector<vector<Real>> rand_mat = random_.RandReal2D(nrandsamples, D);

    for (int k = 0; k < D; k++)
    {
        Real s = 0;
        for(int i = 0; i < task_indices.size(); i++)
        {
            s += pop.at(task_indices[i]).elements[k];
        }
        for(int i = 0; i < nrandsamples; i++)
        {
            s += rand_mat.at(i).at(k);
        }
        mean[k] = s / (nsamples + nrandsamples);
        Real v = 0;
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

Real norm_pdf(Real x, Real m, Real s)
{
    static const Real inv_sqrt_2pi = 0.3989422804014327;
    float a = (x - m) / s;
    Real p = inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
    return p;
}

int update_RMP(vector<Chro> &pop)
{
    vector<tuple<vector<Real>, vector<Real>, int, vector<int>>> probmodels; //mean std nsamples, indices
    // O(m*D*N)
    Real s = get_wall_time();
    for (int i = 0; i < ntasks; i++)
    {
        probmodels.emplace_back(probmodel(pop, i));
    }
    // printf("prob model build time %f\n", get_wall_time() - s);

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
    Real t = r8_epsilon();
    Real e = t;
    Real a = 0;
    Real b = 1;
    Real m = 0;
    Real c = (a + b) / 2;
    // O(m*m*N*D)
    Real s2 = get_wall_time();
    // #pragma omp parallel for 
    for(int n = 0; n < num_items; n++)
    {
        int i = items_to_calc.at(n).first;
        int j = items_to_calc.at(n).second;
        int dims = min(calc_dims.at(i), calc_dims.at(j));

        const vector<Real> &mean_i = std::get<0>(probmodels[i]);
        const vector<Real> &mean_j = std::get<0>(probmodels[j]);

        const vector<Real> &std_i = std::get<1>(probmodels[i]);
        const vector<Real> &std_j = std::get<1>(probmodels[j]);

        int nsamples_i = std::get<2>(probmodels[i]);
        int nsamples_j = std::get<2>(probmodels[j]);

        const vector<int> &indices_i = std::get<3>(probmodels[i]);
        const vector<int> &indices_j = std::get<3>(probmodels[j]);

        probmats[n][0].probmat.resize(nsamples_i, vector<Real>(2, 1.0));
        probmats[n][1].probmat.resize(nsamples_j, vector<Real>(2, 1.0));
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
        Real rmp = 0.0; //init guess
        MyFunc f3 = MyFunc(probmats[n]);
        Real start = get_wall_time();
        Real ret = glomin(a, b, c, m, e, t, f3, rmp);
        RMP[i][j] = max(0.0, rmp + random_.RandRealNormal(0, 0.01));
        RMP[i][j] = min(RMP[i][j], 1.0);
        RMP[j][i] = RMP[i][j];
    }
    // printf("calculate rmp matrix cost time %f\n", get_wall_time() - s2);
    //destroy the probmats
    for(int n = 0; n < num_items; n++)
    {
        delete []probmats[n];
    }
    return 0;
}
/********* Update rmp matrix related code end **********/

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

pair<int, Real> best_in_pop(const vector<Chro> &p, int skill)
{
    Real bestf = DBL_MAX;
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
    int sub_pop_size = args.popsize;
    if (sub_pop_size % 2 != 0) sub_pop_size++; // subpopsize must be even
    D = args.UDim;
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
            c.factorial_costs = vector<Real>(ntasks, DBL_MAX);
            c.factorial_ranks = vector<int>(ntasks, 2*pop_size + 1);
            c.factorial_costs.at(i) = task_evals.at(i)->EvaluateFitness(c.elements);
            c.factorial_ranks.at(i) = j+1;
            pop_all.emplace_back(c);
            evals++;
        }
        // the last individual be the initial best individual.
        best_objs.push_back(std::make_pair(pop_all.back().factorial_costs[i], pop_all.size() - 1));
    }
    if(args.MTO)
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
        Real rmp = RMP[p1.skill_factor][p2.skill_factor];
        vector<Real> cf = EA_solver->generate_cf(D);
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
            /* select another unique individual the same as p1 */
            vector<int> sol1 = find_in_pop(pop_all, p1.skill_factor);
            int num_sol1 = sol1.size();
            int sel1 = sol1[random_.Permutate(num_sol1, 1)[0]];
            while(sel1 == idx_p1)
            {
                sel1 = sol1[random_.Permutate(num_sol1, 1)[0]];
            }
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
    vector<Chro> offspring = variation();
    for (int i = 0; i < offspring.size(); i++)
    {
        Chro &c = offspring.at(i);
        c.factorial_costs = vector<Real>(ntasks, DBL_MAX);
        c.factorial_ranks = vector<int>(ntasks, 2 * pop_size + 1);
        c.factorial_costs[c.skill_factor] = \
            task_evals[c.skill_factor]->EvaluateFitness(c.elements);
    }
    evals += offspring.size();
 
    vector<Chro> inter_pop;
    inter_pop.insert(inter_pop.end(), pop_all.begin(), pop_all.end());
    inter_pop.insert(inter_pop.end(), offspring.begin(), offspring.end());
    int inter_pop_size = inter_pop.size();
    vector<Real> tmp_vec(inter_pop_size, 0.0);
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
        if (args.MTO){
            for(int k = 0; k < ranks.size(); k++)
            {
                if(ranks[k] < min_rank)
                {
                    min_idx = k;
                    min_rank = ranks[k];
                }
            }
        }
        inter_pop[i].skill_factor = min_idx;
        inter_pop[i].scalar_fitness = 1.0 / ranks[min_idx];
        tmp_vec[i] = inter_pop[i].scalar_fitness;
    }

    vector<int> sorted_pop_indices = argsort(tmp_vec);
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
	int generation = 0;
    Real bestf = 0;

	while (generation < args.Gmax)
    {
	    if (args.MTO && MFEA2)
	    {
       	 	update_RMP(pop_all);
	    }
        reproduce();
		if ((generation + 1) % args.record_interval == 0
             || (generation == 0))
        {
			for (int i = 0; i < ntasks; i++)
            {
                Chromosome ind = pop_all.at(best_objs.at(i).second);
                bestf = best_objs.at(i).first;
                fprintf(stderr, "task id %d; runs %d/%d; gens %d/%d; bestf %.12f\n", 
                    i+1, run_id+1, args.total_runs, generation+1, args.Gmax, bestf);
                RecordInfo info;
                info.best_fitness = bestf;
                info.generation = generation+1;
                record_tasks[i].RecordInfos(info);
            }
        }
        generation++;
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
        fprintf(stderr, "task %d run_id %d, final results: [%s]; bestf %.12f \n", i+1, run_id+1, ss.str().c_str(), bestf);
        record_tasks[i].FlushInfos(run_id);
        record_tasks[i].Clear();
    }
    
    uninitialized();

}
int main(int argc, char* argv[])
{
    auto t_start = get_wall_time();
    global_init(argc, argv);
	for (run_id = 0; run_id < args.total_runs; run_id++)
	{
		srand((run_id+1)*10000);
		MFEA();
        cerr << run_id + 1 << " run cost time " << get_wall_time() - t_start
             << " seconds" << endl;
	}
	cerr << "total wall clock time: " << get_wall_time() - t_start 
         << " seconds for " << args.total_runs << " runs" << endl;
	return 0;
}
