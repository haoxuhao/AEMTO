#ifndef __EVALUATOR_H__
#define __EVALUATOR_H__

#include <string>
#include <unordered_map>
#include <functional>
#include "config.h"
#include "Eigen/Dense"


class Evaluator 
{
protected:
    ProblemInfo problem_info_;
public:
    virtual Real EvaluateFitness(const vector<Real> & elements) = 0;
    virtual ~Evaluator() {};
};

class BenchFuncEvaluator : public Evaluator 
{
private: 
    Real scale_rate_;
    Eigen::MatrixXd rM_;
    bool is_rotate_{false};
    Eigen::VectorXd shift_;
    Eigen::VectorXd bias_vec_;
    const unordered_map<string, vector<Real> > func_search_range {
        {"sphere", {-100, 100}}, 
        {"weierstrass", {-0.5, 0.5}},
        {"rosenbrock", {-50, 50}}, 
        {"ackley", {-50, 50}}, 
        {"schwefel", {-500, 500}},
        {"griewank", {-100, 100}}, 
        {"rastrigin", {-50, 50}}
    };
    

    void LoadTaskData();
public:
    BenchFuncEvaluator(const ProblemInfo& probleminfo);
    virtual ~BenchFuncEvaluator(){};
    virtual Real EvaluateFitness(const vector<Real> & elements);
};

class ArmEvaluator : public Evaluator
{
private: 
    vector<Real> fw_kinematics(vector<Real> & commad);
    vector<Real> task;
    int n_dofs;
    Real angular_range;
    vector<Real> lengths;  
    void LoadTaskData();
public:
    ArmEvaluator(const ProblemInfo &problem_info);
    virtual ~ArmEvaluator(){};
    virtual Real EvaluateFitness(const vector<Real> & elements);
};

#endif