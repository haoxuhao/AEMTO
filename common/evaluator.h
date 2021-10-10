#ifndef __EVALUATOR_H__
#define __EVALUATOR_H__

#include "CEC2014.h"
#include <mutex>
#include "config.h"


class Evaluator 
{
public:
    virtual real EvaluateFitness(const vector<real> & elements) = 0;
    virtual int	 Initialize(const ProblemInfo &problem_info) = 0;
	virtual int	 Uninitialize() {return 0;};
};


class BenchFuncEvaluator : public Evaluator, public CEC2014
{
private: 
    int task_id_;
    bool flag_composition_;
    int                     calc_dim_;
    ProblemInfo             problem_info_;
    real                    scale_rate_;
    real                    fixed_shift_;
    int                     rotation_flag_;
    int					    LoadData_matea();
    void                    load_shifts_from_singlefile(string file_name);
    int                     transfer_to_original_space(const vector<real> &elements);
    
public:
    BenchFuncEvaluator(){};
    ~BenchFuncEvaluator(){};
    virtual int				Initialize(const ProblemInfo &problem_info);
	virtual int				Uninitialize();
    virtual real			EvaluateFitness(const vector<real> & elements);
    virtual real			EvaluateFitness(const vector<real> & elements, mutex &mtx);
};

class ArmEvaluator : public Evaluator
{
private: 
    ProblemInfo             problem_info_;
    int					    load_data();
    vector<real>            fw_kinematics(vector<real> & commad);

    vector<real> task;
    int n_dofs;
    real angular_range;
    vector<real> lengths;  

public:
    ArmEvaluator(){};
    ~ArmEvaluator(){};
    virtual int						Initialize(const ProblemInfo &problem_info);
	virtual int						Uninitialize() {return 0;};
    virtual real					EvaluateFitness(const vector<real> & elements);
};
#endif