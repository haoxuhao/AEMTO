#ifndef __RECORD_H__
#define __RECORD_H__
#include <mpi.h>
#include <sstream>
#include <EA.h>
#include <map>
#include "json.hpp"
#include "config.h"
#include "comm.h"


using json = nlohmann::json;

struct RecordInfo
{
    string tag;
    int generation;
    real time;
    real comm_time;
    real best_fitness;
    vector<real> elements;
};

struct ReuseInfo
{
    real best_fitness_update;
    real update_rate_of_self_evolve;
    real update_rate_of_reuse;
    real import_prob;
    unordered_map<int, real> rewards_table;
};

class Record
{
private:
    NodeInfo            node_info_;
    IslandInfo          island_info_;
    ProblemInfo         problem_info_;
    EAInfo              EA_info_;
    string 				file_name_;
    map<int, real>      summed_fes;

    // json for save results of multiple runs
    json                 json_results;
    
public:
						Record(const NodeInfo node_info);
						~Record();
	int                 Initialize(IslandInfo island_info, ProblemInfo problem_info, EAInfo EA_info);
	int                 Uninitialize();

    int                 RecordInfos(RecordInfo &info);
    int                 RecordKnowledgeReuseInfo(real fitness_update,
                                                unordered_map<int, real> &success_insert_rate,
                                                real update_rate_of_self_evolve,
                                                real update_rate_of_reuse);
    int                 RecordKnowledgeReuseInfo(ReuseInfo &info);
    int                 FlushInfos();

    unordered_map<string, real> time_section;
    int transfer_cnt;
 
    int    RECORD_INTERVAL;
    
    // record infos
    vector<real>        record_best_fitnesses;
    vector<int>         record_generations;
    vector<real>        record_time_points;
    vector<vector<real>>    record_solutions;
    vector<real>        record_best_fitness_update_after_reuse;
    vector<unordered_map<int, real>> record_success_insert_rate_in_reuse;
    vector<real>        record_update_rate_of_self_evolve;
    vector<real>        record_update_rate_of_reuse;
    vector<real>        record_import_prob;
    vector<vector<real>> recored_selection_probs;
    vector<real> success_offsprings_rate;
    vector<real> success_best_update_rate;
    vector<pair<int, real>>        record_mean_fitness;
};

#endif
