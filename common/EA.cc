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

int EA::InitializePopulation(Population & population)
{
    for(int i = 0; i < island_info_.island_size; i++)
    {
        Individual tmp_individual;
        for (int j = 0; j < problem_info_.dim; j++)
            tmp_individual.elements.push_back(random_.RandRealUnif(problem_info_.min_bound, problem_info_.max_bound));
        tmp_individual.fitness_value = func->EvaluateFitness(tmp_individual.elements);
        tmp_individual.skill_factor = -1;
        population.push_back(tmp_individual);
    }
    return 0;
}

int EA::InitializePopulation(Population &pop, unique_ptr<FuncEval> &eval)
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

int EA::Uninitialize()
{
    return 0;
}
real EA::CheckBound(real to_check_elements, real min_bound, real max_bound)
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

Population EA::EvaluatePop(Population &p, unique_ptr<FuncEval> &eval)
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

real EA::PopImprovement(Population &pop_curr, Population &pop_pre)
{
    SortPop(pop_curr);
    SortPop(pop_pre);
    real res = 0;
    int pop_size = (int)pop_curr.size();
    real delta = pop_curr[0].fitness_value;
    for (int i = 0; i < pop_size; i++)
    {
        res += delta / pop_curr[i].fitness_value * \
            (pop_pre[i].fitness_value - pop_curr[i].fitness_value);
    }
    res /= (real) pop_size;
    res /= delta;
    return res;
}
real check_bnd(real x, real lb, real ub)
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

Individual binomial_crossover(const Individual &p1, const Individual &p2, real cr)
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