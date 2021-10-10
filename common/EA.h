#ifndef __H_EA_H__
#define __H_EA_H__

#include "config.h"
#include <stdio.h>
#include <sstream>
#include "evaluator.h"
#include "random.h"

struct GAParam
{
	real mum;
	real mu;
	real probswap;
};

struct DEInfo
{
    real CR;
    real F;
	real LCR;
	real UCR;
	real LKTCR;
	real UKTCR;
	real LF;
	real UF;
    int strategy_ID;
    int group_size;
	int group_num;
	real ktc_cr;
	real transfer_cross_over_rate;
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


typedef Evaluator FuncEval;

class EA
{
protected:
    ProblemInfo             problem_info_;
    IslandInfo              island_info_;
    NodeInfo                node_info_;
    EAInfo                  EA_info_;
    Random                  random_;
    FuncEval *func;
public:
                            EA();
                            ~EA();
    virtual int             InitializePopulation(Population & population);
    virtual int             Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info);
    virtual int             Uninitialize();
    int                     InitializePopulation(Population &pop, unique_ptr<FuncEval> &func_eval);
    real                    CheckBound(real to_check_elements, real min_bound, real max_bound);
    Individual              FindBestIndividual(Population & population);
    virtual string          GetParameters(DEInfo DE_info)=0;
    virtual real            Run(Population & population)=0;
    virtual int             ConfigureEA(EAInfo EA_info)=0;
    virtual Population      EvaluatePop(Population &p, unique_ptr<FuncEval> &eval);
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
    virtual real            PopImprovement(Population &pop_curr, Population &pop_pre);

    void set_func(Evaluator *func) {this->func = func;}
};

class DE_CPU : public EA
{
protected:
    DEInfo                  DE_info_;
    int                     Reproduce(Population & population);
    
public:
                            DE_CPU(){};
                            DE_CPU(NodeInfo node_info);
                            ~DE_CPU();
    virtual int             Initialize(IslandInfo island_info, ProblemInfo problem_info, DEInfo DE_info);
    virtual int             InitializePopulation(Population & population);
    virtual int             Uninitialize();
    virtual real            Run(Population & population);
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
        int                     Reproduce(Population &pop, unique_ptr<FuncEval> &eval_func);
        virtual string          GetParameters(DEInfo DE_info);
        virtual real             Run(Population & population);
        virtual int             ConfigureEA(EAInfo EA_info);
        Individual              crossover(const Individual &p1, const Individual &p2, const vector<real> &cf);
        Individual              mutate(Individual &p);

        vector<real>            crossover(const vector<real> &p1, const vector<real> &p2, const vector<real> &cf);
        vector<real>            mutate(vector<real> &p);
        int                     swap(vector<real> &p1, vector<real> &p2);
        vector<real>            generate_cf(int);
};

Individual binomial_crossover(const Individual &p1, const Individual &p2, real cr);
real check_bnd(real x, real lb, real ub);

#endif