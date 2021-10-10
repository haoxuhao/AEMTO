#include <algorithm>

#include "EA.h"
#include "util.h"

/*
    SBX
*/
vector<real> GA_CPU::crossover(const vector<real> &p1,
                               const vector<real> &p2,
                               const vector<real> &cf)
{
    int dim = p1.size();
    vector<real> ret;
    ret.resize(dim, 0.0);
    for (int i = 0; i < dim; i++)
    {
        ret[i] = 0.5 * ((1 + cf[i]) * p1[i] + (1 - cf[i]) * p2[i]);
        ret[i] = min(1.0f, (float)ret[i]);
        ret[i] = max(0.0f, (float)ret[i]);
    }
    return ret; 
}
Individual GA_CPU::crossover(const Individual &p1, const Individual &p2, const vector<real> &cf)
{
    Individual ret;
    ret.elements = crossover(p1.elements, p2.elements, cf); 
    return ret;
}

/*
    polynomial mutation
*/
vector<real> GA_CPU::mutate(vector<real> &p)
{
    real mum = EA_info_.ga_param.mum;
    int dim = (int)p.size();
    for (int i = 0; i < dim; i++)
    {
        if(random_.RandRealUnif(0, 1) < (1.0 / dim))
        {
            real u = random_.RandRealUnif(0, 1);
            if (u <= 0.5)
            {
                real d = pow((2 * u), 1.0 / (1.0 + mum)) - 1;
                p[i] = p[i] + d * p[i];
            }
            else
            {
                real d = 1 - pow((2 * (1 - u)), 1.0 / (1.0 + mum));
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
vector<real> GA_CPU::generate_cf(int dim)
{
    vector<real> cf(dim, 0.0);
    real mu = EA_info_.ga_param.mu;
    for (int k = 0; k < dim; k++)
    {
        real u = random_.RandRealUnif(0, 1);
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

int GA_CPU::swap(vector<real> &c1, vector<real> &c2)
{
    real probswap = EA_info_.ga_param.probswap;
    int dim = c1.size();
    for(int k = 0; k < dim; k++)
    {
        if (random_.RandRealUnif(0, 1) >= probswap)
        {
            real tmp = c2[k];
            c2[k] = c1[k];
            c1[k] = tmp;
        }
    }
}
Population GA_CPU::Variation(Population &pop)
{
    int pop_size = (int)pop.size();
    int dim = (int)pop[0].elements.size();
    real probswap = EA_info_.ga_param.probswap;
    Population offsp;
    vector<int> rand_indices = random_.Permutate(pop_size, pop_size);
    for (int i = 0; i < pop_size / 2; i++)
    {
        Individual p1 = pop[rand_indices[i]];
        Individual p2 = pop[rand_indices[i + pop_size / 2 - 1]];
        
        vector<real> cf = generate_cf(dim);
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

int GA_CPU::Reproduce(Population &pop, unique_ptr<FuncEval> &eval_func)
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
real GA_CPU::Run(Population & pop)
{
    // TO DO here, func is pointer
    // return Reproduce(pop, func) / (real) island_info_.island_size;
}
int GA_CPU::ConfigureEA(EAInfo EA_info)
{
    return 0;
}