#ifndef __H_EA_H__
#define __H_EA_H__

#include "config.h"
#include <stdio.h>
#include <sstream>
#include "evaluator.h"
#include "random.h"

struct GAParam
{
	Real mum;
	Real mu;
	Real probswap;
};

struct EAInfo
{
    Real CR;
    Real F;
	Real LKTCR;
	Real UKTCR;
	string STO;
	GAParam ga_param;
};

/**
 * Evolve rewards
 * Generated in current survival selection
 */
struct EvolveRewards
{
	int update_num;
};

class EA
{
protected:
    ProblemInfo             problem_info_;
    EAInfo                  EA_info_;
    Random                  random_;
public:
    EA(const ProblemInfo &problem_info, const EAInfo ea_info) 
        : problem_info_(problem_info), EA_info_(ea_info) {};
    int                     InitializePopulation(Population &pop, unique_ptr<Evaluator> &func_eval);
    Real                    CheckBound(Real to_check_elements, Real min_bound, Real max_bound);
    Individual              FindBestIndividual(Population & population);
    virtual Real            Run(Population & population, unique_ptr<Evaluator> &eval)=0;
    virtual Population      EvaluatePop(Population &p, unique_ptr<Evaluator> &eval);
    virtual Population      Survival(Population &pop, Population &offsp, EvolveRewards &out);
    /**
     * Populaiton sort according to fitness values
     * Inplace sort
     * @param pop to sort
     * @param pop sorted
     */
    Population              SortPop(Population &pop, bool ascent=true);
    /**
     * Generate offspring
     */
    virtual Population      Variation(Population &pop) = 0;
};

class DE : public EA
{
protected:
    int                     Reproduce(Population & population);
    int                     ReproduceV2(Population & population, unique_ptr<Evaluator> &eval);
    
public:
    DE(const ProblemInfo &problem_info, const EAInfo ea_info) 
        : EA(problem_info, ea_info) {};
    virtual Real            Run(Population & population, unique_ptr<Evaluator> &eval);
    virtual Population      Variation(Population &pop);
    virtual Population      Survival(Population & population, Population &offsp, EvolveRewards &out);
};

class GA : public EA
{
    public:
        GA(const ProblemInfo &problem_info, const EAInfo ea_info) 
            : EA(problem_info, ea_info) {};
        /**
         * Generate offspring
         */
        virtual Population      Variation(Population &pop);

        /**
         * Evolve a new population
         * Return update num
         * The same as MFEA2 did.
         */
        int                     Reproduce(Population &pop, unique_ptr<Evaluator> &eval_func);
        virtual Real            Run(Population & population, unique_ptr<Evaluator> &eval) {return 0;};
        Individual              crossover(const Individual &p1, const Individual &p2, const vector<Real> &cf);
        Individual              mutate(Individual &p);
        vector<Real>            crossover(const vector<Real> &p1, const vector<Real> &p2, const vector<Real> &cf);
        vector<Real>            mutate(vector<Real> &p);
        int                     swap(vector<Real> &p1, vector<Real> &p2);
        vector<Real>            generate_cf(int);
};

Individual binomial_crossover(const Individual &p1, const Individual &p2, Real cr);

#endif