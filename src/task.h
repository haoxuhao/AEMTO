#ifndef __H_TASK_H__
#define __H_TASK_H__
#include <future>
#include "config.h"
#include "EA.h"
#include "communicator.h"
#include "migrate.h"
#include "record.h"
#include "knowledge_reuse.h"
#include "evaluator.h"

struct TimeSec {
    real total_time = 0;
    real total_kt_time = 0;
    real comm_time = 0;
    real wait_time = 0;
    real EA_time = 0; 
};

class Task
{
    public:
        Task(NodeInfo &node_info, ProblemInfo &problem_info, IslandInfo &island_info, EAInfo &EA_info);
        ~Task();
        int TaskInitialize(int run_id=1);
        int TaskUnInitialize();
        Population prepare_emigrations()
        {
            return migrate.PrepareEmigrations(population);
        };
        Individual find_best_individual()
        {
            return EA_solver->FindBestIndividual(population);
        };
        real Reuse(Population &other, unordered_map<int, int> &ns);
        real EA_solve();
        unique_ptr<EA> EA_solver;
        Communicator comm;
        KnowledgeReuse knowledge_reuse;
        Evaluator* func;
        Random random_;
        
        ProblemInfo problem_info;
        IslandInfo island_info;
        EAInfo EA_info;
        NodeInfo node_info;
        Population population;
        Migrate migrate;
        Record record;
        long max_gens;
        long record_gens;
        real bestf;

        TimeSec time_record;
        int best_update_times;
        real individuals_update; 
        
        void update_bestf() {bestf = find_best_individual().fitness_value;}
        real GetBestFitness() {return bestf;}
        Population & GetPop() {return population;};
        bool convergened(){return abs(GetBestFitness()) < 1e-10;};
        void step(int gen);
};
void knowledge_provider(
        std::future<void> future, vector<shared_ptr<Task>> &tasks, int tid=0);

void knowledge_request(future<void> fut, 
        vector<shared_ptr<Task>> tasks, int curr_task_id,
        Population &recv_pop);

int iterate(vector<shared_ptr<Task>> tasks, int task_id, int run_id, int gen);

void assign_tasks_to_node(vector<int> &task_ids, NodeInfo &node_info);
#endif 