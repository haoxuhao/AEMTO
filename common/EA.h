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

struct DEInfo
{
    Real CR;
    Real F;
	Real LCR;
	Real UCR;
	Real LKTCR;
	Real UKTCR;
	Real LF;
	Real UF;
    int strategy_ID;
    int group_size;
	int group_num;
	Real ktc_cr;
	Real transfer_cross_over_rate;
	string EA_parameters;
	string STO;
	GAParam ga_param;
	string ktcr_strategy;
};
typedef DEInfo EAInfo;

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
    IslandInfo              island_info_;
    EAInfo                  EA_info_;
    Random                  random_;
public:
                            EA();
                            ~EA();
    virtual int             Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info);
    int                     InitializePopulation(Population &pop, unique_ptr<Evaluator> &func_eval);
    Real                    CheckBound(Real to_check_elements, Real min_bound, Real max_bound);
    Individual              FindBestIndividual(Population & population);
    virtual string          GetParameters(DEInfo DE_info)=0;
    virtual Real            Run(Population & population, unique_ptr<Evaluator> &eval)=0;
    virtual int             ConfigureEA(EAInfo EA_info)=0;
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
    /**
     * Population improvement
     * Calculate the population improvement
     * Relative improvement as default setting
    */
    virtual Real            PopImprovement(Population &pop_curr, Population &pop_pre);
};

class DE_CPU : public EA
{
protected:
    DEInfo                  DE_info_;
    int                     Reproduce(Population & population);
    int                     ReproduceV2(Population & population, unique_ptr<Evaluator> &eval);
    
public:
                            DE_CPU(){};
                            ~DE_CPU();
    virtual int             Initialize(IslandInfo island_info, ProblemInfo problem_info, DEInfo DE_info);
    virtual Real            Run(Population & population, unique_ptr<Evaluator> &eval);
    virtual string          GetParameters(DEInfo DE_info);
    virtual int             ConfigureEA(DEInfo DE_info);
    virtual Population      Variation(Population &pop);
    virtual Population      Survival(Population & population, Population &offsp, EvolveRewards &out);
};

class GA_CPU : public EA
{
    public:
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
        virtual string          GetParameters(DEInfo DE_info);
        virtual Real            Run(Population & population, unique_ptr<Evaluator> &eval) {return 0;};
        virtual int             ConfigureEA(EAInfo EA_info);
        Individual              crossover(const Individual &p1, const Individual &p2, const vector<Real> &cf);
        Individual              mutate(Individual &p);

        vector<Real>            crossover(const vector<Real> &p1, const vector<Real> &p2, const vector<Real> &cf);
        vector<Real>            mutate(vector<Real> &p);
        int                     swap(vector<Real> &p1, vector<Real> &p2);
        vector<Real>            generate_cf(int);
};

Individual binomial_crossover(const Individual &p1, const Individual &p2, Real cr);
Real check_bnd(Real x, Real lb, Real ub);

#endif