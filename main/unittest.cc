#include <iostream>
#include <algorithm>
#include <unordered_map>
#include "config.h"
#include "random.h"
#include "util.h"
#include "evaluator.h"


using namespace std;


void arm_eval_test()
{
    ProblemInfo problem_info;
    ArmEvaluator arm(problem_info);
    problem_info.shift_data_file = "/fred/oz121/hxu/2020_mouret_gecco/src";
    problem_info.dim = 10;
    problem_info.task_id = 1;
    vector<Real> elems = {0.884907, 0.344362, 0.244923, 0.0370074, 0.920197, 0.420731, 0.540009, 0.818535, 0.874266, 0.299411};
    Real fitness = arm.EvaluateFitness(elems);
    printf("fitness %f\n", fitness);
}

void matrix_test()
{
    Mat a = {{1, 2}, {1, 2}};
    Mat b = {{2, 1}, {2, 1}};
    Mat c = matrix_multiply(a, b);
    for (const auto &row : c)
    {
        for (const auto &e : row)
        {
            cout << e << ", ";
        }
        cout << endl;
    }
    Mat d = matrix_transpose(c);
    for (const auto &row : d)
    {
        for (const auto &e : row)
        {
            cout << e << ", ";
        }
        cout << endl;
    } 
}

int main()
{
    /*roulette_sampling test*/
    // Random random_;
    // //c++11
    // vector<Real> pdf = {0.18, 0.16, 0.15, 0.13, 0.11, 0.09, 0.07, 0.06, 0.03, 0.02, 0.0};
    // cout << "pdf: ";
    // for(auto e : pdf)
    // {
    //     cout << e << ", ";
    // }
    // cout << endl;

    // for(int i = 0; i < 20; i++)
    // {
    //     cout << "run " << i << " " << random_.roulette_sampling(pdf) << endl;      
    // }
    
    // /* find median of a vector */
    // assert(median({5, 5, 5, 0, 0, 0, 1, 2}) == 1.5 && "median test error");
    // assert(median({1, 2, 3}) == 2 && "median test error");

    // /* variable vector */
    // vector<vector<Real> > mem;
    // mem.push_back({1,2,3});
    // mem.push_back({4,5});
    // for(auto e : mem)
    // {
    //     for (auto x : e)
    //     {
    //         cout << x << " ";
    //     }
    //     cout << endl;
    // }

    // /*roulette sample fitnesses*/
    // vector<Real> fitnesses = {0.18, 0.16, 0.15, 0.13, 0.11, 0.09, 0.07, 0.06, 0.03, 0.02, 0.1};
    // cout << "fitnesses: ";
    // for(auto e : fitnesses)
    // {
    //     cout << e << ", ";
    // }
    // cout << endl;
    // vector<int> rouletee_indices = random_.roulette_sample(fitnesses, 7);
    // for(const auto &e : rouletee_indices){
    //     cout << "sample " << e << " value: " << fitnesses[e] << endl;
    // }

    //============================
    // matrix_test();
    //============================
    // arm_eval_test();

    // sus sample test
    // vector<Real> pdf = {0.18, 0.16, 0.15, 0.13, 0.11, 0.09, 0.13, 0.05};
    vector<Real> pdf = {0.18, 0.16, 0.15, 0.13, 0.11, 0.09, 0.07, 0.06, 0.03, 0.02, 0.1};
    Real pdf_sum = accumulate(pdf.begin(), pdf.end(), 0.0);
    std::for_each(pdf.begin(), pdf.end(), [&](Real &x) {x /= pdf_sum;});
    int sele_num = 20;
    Random rand_;
    vector<int> sel_indices = rand_.sus_sample(pdf, sele_num);
    unordered_map<int, int> task_selected_times;
    for(int i = 0; i < sel_indices.size(); i++) {
        cout << "pointer " << i << ", index " << sel_indices[i] << endl;
        if (task_selected_times.find(sel_indices[i]) == task_selected_times.end()) {
            task_selected_times[sel_indices[i]] = 1;
        } else {
            task_selected_times[sel_indices[i]]++;
        }
    }
    for (const auto & e: task_selected_times) {
        cout << "random res: key " << e.first << ", val " << e.second << endl;
    }

    return 0;
}