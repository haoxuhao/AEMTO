#include <algorithm>
#include "EA.h"
#include "util.h"

EA::EA()
{

}

EA::~EA()
{

}

int EA::Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info)
{
    problem_info_ = problem_info;   
    island_info_ = island_info;
    EA_info_ = EA_info;
    return 0;
}

int EA::InitializePopulation(Population &pop, unique_ptr<Evaluator> &eval)
{
   for(int i = 0; i < island_info_.island_size; i++)
    {
        Individual tmp_individual;
        for (int j = 0; j < problem_info_.dim; j++)
            tmp_individual.elements.push_back(random_.RandRealUnif(problem_info_.min_bound, problem_info_.max_bound));
        tmp_individual.fitness_value = eval->EvaluateFitness(tmp_individual.elements);
        tmp_individual.skill_factor = -1;
        pop.push_back(tmp_individual);
    }
    return 0; 
}

Real EA::CheckBound(Real to_check_elements, Real min_bound, Real max_bound)
{
	while ((to_check_elements < min_bound) || (to_check_elements > max_bound))
	{
		if (to_check_elements < min_bound)
			to_check_elements = min_bound + (min_bound - to_check_elements);
		if (to_check_elements > max_bound)
			to_check_elements = max_bound - (to_check_elements - max_bound);
	}
	return to_check_elements;
}

Individual EA::FindBestIndividual(Population & population)
{
    int best_individual_ind = 0;
    double best_individual_fitness_value = population[0].fitness_value;
    for(int i = 1; i < island_info_.island_size; i++)
    {
        if(population[i].fitness_value < best_individual_fitness_value)
        {
            best_individual_ind = i;
            best_individual_fitness_value = population[i].fitness_value;
        }
    }
    return population[best_individual_ind];
}

Population EA::EvaluatePop(Population &p, unique_ptr<Evaluator> &eval)
{
    for (auto &i : p)
    {
        i.fitness_value = eval->EvaluateFitness(i.elements);
    }
    return p;
}

Population EA::SortPop(Population &pop, bool ascent)
{
    sort(pop.begin(), pop.end(), 
        [](const Individual &a, const Individual &b){
            return a.fitness_value < b.fitness_value;
    });
    return pop;
}

// do elitism selection
Population EA::Survival(Population &pop, Population &offsp, EvolveRewards &out)
{
    int pop_size = (int)pop.size();
    vector<Individual> inter_pop;
    inter_pop.insert(inter_pop.end(), pop.begin(), pop.end());
    inter_pop.insert(inter_pop.end(), offsp.begin(), offsp.end());
    
    int update_num = 0;
    vector<int> sorted_indices = argsort_population(inter_pop);
    for (int i = 0; i < pop_size; i++)
    {
        pop[i] = inter_pop[sorted_indices[i]];
        pop[i].skill_factor = -1;
        if (sorted_indices[i] >= pop.size())
        {
            update_num++;
        }
    }
    out.update_num = update_num;
    return pop;
}

Real EA::PopImprovement(Population &pop_curr, Population &pop_pre)
{
    SortPop(pop_curr);
    SortPop(pop_pre);
    Real res = 0;
    int pop_size = (int)pop_curr.size();
    Real delta = pop_curr[0].fitness_value;
    for (int i = 0; i < pop_size; i++)
    {
        res += delta / pop_curr[i].fitness_value * \
            (pop_pre[i].fitness_value - pop_curr[i].fitness_value);
    }
    res /= (Real) pop_size;
    res /= delta;
    return res;
}
Real check_bnd(Real x, Real lb, Real ub)
{
    while ((x < lb) || (x > ub))
	{
		if (x < lb)
			x = lb + (lb - x);
		if (x > ub)
			x = ub - (x - ub);
	}
 
	return x;
}

Individual binomial_crossover(const Individual &p1, const Individual &p2, Real cr)
{
    assert(p1.elements.size() == p2.elements.size() &&
           "same dimension of p1 and p2 required.");
    int dim = p1.elements.size();
    Individual c;
    c.elements = p1.elements;
    Random rand;
    int j = rand.RandIntUnif(0, dim - 1);
    for (int k = 0; k < dim; k++)
    {
        if(rand.RandRealUnif(0.0, 1.0) <= cr || k == j)
        {
            c.elements[k] = p2.elements[k];
        }
    }
    return c;
}
/*
    SBX
*/
vector<Real> GA_CPU::crossover(const vector<Real> &p1,
                               const vector<Real> &p2,
                               const vector<Real> &cf)
{
    int dim = p1.size();
    vector<Real> ret;
    ret.resize(dim, 0.0);
    for (int i = 0; i < dim; i++)
    {
        ret[i] = 0.5 * ((1 + cf[i]) * p1[i] + (1 - cf[i]) * p2[i]);
        ret[i] = min(1.0f, (float)ret[i]);
        ret[i] = max(0.0f, (float)ret[i]);
    }
    return ret; 
}
Individual GA_CPU::crossover(const Individual &p1, const Individual &p2, const vector<Real> &cf)
{
    Individual ret;
    ret.elements = crossover(p1.elements, p2.elements, cf); 
    return ret;
}

/*
    polynomial mutation
*/
vector<Real> GA_CPU::mutate(vector<Real> &p)
{
    Real mum = EA_info_.ga_param.mum;
    int dim = (int)p.size();
    for (int i = 0; i < dim; i++)
    {
        if(random_.RandRealUnif(0, 1) < (1.0 / dim))
        {
            Real u = random_.RandRealUnif(0, 1);
            if (u <= 0.5)
            {
                Real d = pow((2 * u), 1.0 / (1.0 + mum)) - 1;
                p[i] = p[i] + d * p[i];
            }
            else
            {
                Real d = 1 - pow((2 * (1 - u)), 1.0 / (1.0 + mum));
                p[i] = p[i] + d * (1.0 - p[i]); 
            }
        }
    }
    return p; 
}
Individual GA_CPU::mutate(Individual &p)
{
    mutate(p.elements);
    return p;
}

/**
 *  params prepare
 */
vector<Real> GA_CPU::generate_cf(int dim)
{
    vector<Real> cf(dim, 0.0);
    Real mu = EA_info_.ga_param.mu;
    for (int k = 0; k < dim; k++)
    {
        Real u = random_.RandRealUnif(0, 1);
        if(u <= 0.5)
        {
            cf[k] = pow(2 * u, (1.0 / (mu + 1)));
        }
        else
        {
            cf[k] = pow(2 * (1 - u), (-1.0 / (mu + 1)));
        }
    }
    return cf;
}

int GA_CPU::swap(vector<Real> &c1, vector<Real> &c2)
{
    Real probswap = EA_info_.ga_param.probswap;
    int dim = c1.size();
    for(int k = 0; k < dim; k++)
    {
        if (random_.RandRealUnif(0, 1) >= probswap)
        {
            Real tmp = c2[k];
            c2[k] = c1[k];
            c1[k] = tmp;
        }
    }
}
Population GA_CPU::Variation(Population &pop)
{
    int pop_size = (int)pop.size();
    int dim = (int)pop[0].elements.size();
    Real probswap = EA_info_.ga_param.probswap;
    Population offsp;
    vector<int> rand_indices = random_.Permutate(pop_size, pop_size);
    for (int i = 0; i < pop_size / 2; i++)
    {
        Individual p1 = pop[rand_indices[i]];
        Individual p2 = pop[rand_indices[i + pop_size / 2 - 1]];
        
        vector<Real> cf = generate_cf(dim);
        // crossover
        Individual c1 = crossover(p1, p2, cf);
        Individual c2 = crossover(p2, p1, cf);
        // mutate
        if(random_.RandRealUnif(0, 1) < 1)
        {
            c1 = mutate(c1);
            c2 = mutate(c2);
        }
        swap(c1.elements, c2.elements);
        c1.skill_factor = -1;
        c2.skill_factor = -1;
        offsp.push_back(c1);
        offsp.push_back(c2);
    }
    return offsp;
}

int GA_CPU::Reproduce(Population &pop, unique_ptr<Evaluator> &eval_func)
{
    Population offsp = Variation(pop);
    offsp = EvaluatePop(offsp, eval_func);
    EvolveRewards rewards;
    pop = Survival(pop, offsp, rewards);
    return rewards.update_num;
}

string GA_CPU::GetParameters(DEInfo DE_info)
{
    return "default GA SBX + Polynomial mutation";
}
int GA_CPU::ConfigureEA(EAInfo EA_info)
{
    return 0;
}

DE_CPU::~DE_CPU()
{
}

string DE_CPU::GetParameters(DEInfo DE_info)
{
    stringstream ss;
    ss << "DE/rand/1/bin";
    return ss.str();
}

int DE_CPU::Initialize(IslandInfo island_info, ProblemInfo problem_info, DEInfo DE_info)
{
	EA::Initialize(island_info, problem_info, DE_info);
    DE_info_ = DE_info;
	return 0;
}
int DE_CPU::ConfigureEA(DEInfo DE_info)
{
    DE_info_ = DE_info;
    return 0;
}

Population DE_CPU::Variation(Population &population)
{
    Real F = DE_info_.F;
    Real CR = DE_info_.CR;
    Population offsp;
    for (int i = 0; i < island_info_.island_size; i++)
    {
        Individual tmp_individual = population[i];
        vector<int> r = random_.Permutate(island_info_.island_size, 5);
        int k = random_.RandIntUnif(0, problem_info_.dim - 1);
        if(DE_info_.strategy_ID == 9)
        {
            CR = random_.RandRealUnif(DE_info_.LCR, DE_info_.UCR);
            F = random_.RandRealUnif(DE_info_.LF, DE_info_.UF);
        }
        for (int j = 0; j < problem_info_.dim; j++)
        {
            tmp_individual.elements[j] = population[r[0]].elements[j] + F * (population[r[1]].elements[j] - population[r[2]].elements[j]);
            if (random_.RandRealUnif(0, 1) > CR && j != k)
                tmp_individual.elements[j] = population[i].elements[j];
            tmp_individual.elements[j] = CheckBound(tmp_individual.elements[j], problem_info_.min_bound, problem_info_.max_bound);
        }
        tmp_individual.skill_factor = -1;
        offsp.emplace_back(tmp_individual);
    }
    return offsp;
}

Population DE_CPU::Survival(Population &pop, Population &offsp, EvolveRewards &out)
{
    assert(pop.size() == offsp.size() && \
        "size of offspring must be equal to population in DE.");
    int update_num = 0;
    for (int i = 0; i < pop.size(); i++)
    {
        if(pop[i].fitness_value > offsp[i].fitness_value)
        {
            pop[i] = offsp[i];
            update_num++;
        }
    }
    out.update_num = update_num;
    return pop;
}

Real DE_CPU::Run(Population & population, unique_ptr<Evaluator> &eval)
{
    int update_num = ReproduceV2(population, eval);
    return update_num/(Real)island_info_.island_size;
}
int DE_CPU::ReproduceV2(Population & population, unique_ptr<Evaluator> &eval)
{
    Real F = DE_info_.F;
    Real CR = DE_info_.CR;
    int update_num = 0;
    for (int i = 0; i < island_info_.island_size; i++)
    {
        Individual tmp_individual = population[i];
        vector<int> r = random_.Permutate(island_info_.island_size, 5);
        int k = random_.RandIntUnif(0, problem_info_.dim - 1);

        for (int j = 0; j < problem_info_.dim; j++)
        {
            if (j == k || random_.RandRealUnif(0, 1) < CR)
            {
                tmp_individual.elements[j] = population[r[0]].elements[j] + F * (population[r[1]].elements[j] - population[r[2]].elements[j]);    
            }
            tmp_individual.elements[j] = CheckBound(tmp_individual.elements[j], problem_info_.min_bound, problem_info_.max_bound);
        }
        tmp_individual.fitness_value = eval->EvaluateFitness(tmp_individual.elements);
        if (tmp_individual.fitness_value < population[i].fitness_value)
        {
            population[i].fitness_value = tmp_individual.fitness_value;
            population[i].elements = tmp_individual.elements;
            update_num++;
        }
    }
    return update_num;
}
