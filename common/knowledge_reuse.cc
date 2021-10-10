#include "knowledge_reuse.h"
#include "EA.h"


real KnowledgeReuse::Reuse(
        Population &pop, Population &other_pop, 
        unordered_map<int, int> &success_insert_table)
{
    int other_pop_size = other_pop.size();
    if(other_pop_size == 0) return 0.0;
    vector<int> rand_indexs = random_.Permutate(other_pop_size, other_pop_size);
    int N = pop.size();
    real update_num = 0;
    for (int k = 0; k < N; k++)
    {
        Individual mu = other_pop.at(rand_indexs[k % other_pop_size]);
        Individual x = pop.at(k);
        real cr = random_.RandRealUnif(ea_info_.LKTCR, ea_info_.UKTCR);
        Individual c = binomial_crossover(x, mu, cr);
        c.fitness_value = func_eval_->EvaluateFitness(c.elements);
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
