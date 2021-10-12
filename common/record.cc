#include "record.h"
#include "util.h"
#include <fstream>

Record::Record(const NodeInfo node_info)
{
    node_info_ = node_info;
    json_results = json::array();
}

Record::~Record() { }

int Record::Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info)
{
    island_info_ = island_info;
    problem_info_ = problem_info;
    EA_info_ = EA_info;
    RECORD_INTERVAL = island_info_.print_interval;
    return 0;
}
int Record::Uninitialize()
{
    record_generations.clear();
    record_solutions.clear();
    record_time_points.clear();
    record_best_fitnesses.clear();
    record_best_fitness_update_after_reuse.clear();
    record_success_insert_rate_in_reuse.clear();
    record_update_rate_of_reuse.clear();
    record_update_rate_of_self_evolve.clear();
    record_import_prob.clear();
    recored_selection_probs.clear();
    success_best_update_rate.clear();
    success_offsprings_rate.clear();
    time_section.clear();
    record_mean_fitness.clear();

    return 0;
}

int Record::FlushInfos()
{
    json res;
    res["run_id"] = problem_info_.run_ID;
    res["fitness_values"] = record_best_fitnesses;
    res["generations"] = record_generations;
    if (record_solutions.size() > 0){
        res["best_solutions"] = record_solutions[record_solutions.size()-1];
    }
    res["best_fitness_update_after_reuse"] = record_best_fitness_update_after_reuse;
    res["success_insert_rate_in_reuse"] = record_success_insert_rate_in_reuse;
    res["update_rate_of_self_evolve"] = record_update_rate_of_self_evolve;
    res["update_rate_of_reuse"] = record_update_rate_of_reuse;
    res["import_prob"] = record_import_prob;
    res["time"] = time_section;
    res["transfer_cnt"] = transfer_cnt;
    res["selection_probs"] = recored_selection_probs;
    res["success_best_update_rate"] = success_best_update_rate;
    res["success_offsprings_rate"] = success_offsprings_rate;
    res["mean_fitness"] = record_mean_fitness;
    
    json_results.emplace_back(res);
    string save_file = island_info_.results_dir+ "/" + island_info_.results_subdir + "/res_task_" + std::to_string(problem_info_.task_id) + ".json";
    ofstream ofs(save_file, ios::out); 
    ofs << json_results.dump() << endl;
    return 0;
}

int Record::RecordInfos(RecordInfo &info)
{
    record_generations.emplace_back(info.generation);
    record_solutions.emplace_back(info.elements);
    record_best_fitnesses.emplace_back(info.best_fitness);
    return 0;
}

int Record::RecordKnowledgeReuseInfo(
    real fitness_update, 
    unordered_map<int, real>& success_insert_rate, 
    real update_rate_of_self_evolve, 
    real update_rate_of_reuse)
{
    record_best_fitness_update_after_reuse.push_back(fitness_update);
    record_success_insert_rate_in_reuse.emplace_back(success_insert_rate);
    record_update_rate_of_self_evolve.emplace_back(update_rate_of_self_evolve);
    record_update_rate_of_reuse.emplace_back(update_rate_of_reuse);

    return 0;
}

int Record::RecordKnowledgeReuseInfo(ReuseInfo &info)
{
    record_best_fitness_update_after_reuse.push_back(info.best_fitness_update);
    record_success_insert_rate_in_reuse.emplace_back(info.rewards_table);
    record_update_rate_of_self_evolve.emplace_back(info.update_rate_of_self_evolve);
    record_update_rate_of_reuse.emplace_back(info.update_rate_of_reuse);
    // record_import_prob.emplace_back(info.import_prob);
    return 0;
}
