#ifndef __RECORD_H__
#define __RECORD_H__
#include "json.hpp"
#include "config.h"

using json = nlohmann::json;

struct RecordInfo
{
    string tag;
    int generation;
    Real best_fitness;
    vector<Real> elements;
};

class Record
{
private:
    json json_results;
    Args args_;
    int task_id_{0};
    string save_file;
    
public:
	Record(const Args &args, int task_id): args_(args), task_id_(task_id) {
        json_results = json::array();
        save_file = args_.results_dir + 
                    "/res_task_" + std::to_string(task_id_) + ".json";
    };
	~Record() {};
    void RecordInfos(RecordInfo &info) {
        record_generations.emplace_back(info.generation);
        record_solutions.emplace_back(info.elements);
        record_best_fitnesses.emplace_back(info.best_fitness);
    };
    int FlushInfos(int run_id = 0) {
        json res;
        res["run_id"] = run_id;
        res["fitness_values"] = record_best_fitnesses;
        res["generations"] = record_generations;
        if (record_solutions.size() > 0){
            res["best_solutions"] = record_solutions[record_solutions.size()-1];
        } 
        json_results.emplace_back(res);
        ofstream ofs(save_file, ios::out); 
        ofs << json_results.dump() << endl;
        return 0;
    };
    void Clear() {
        record_generations.clear();
        record_solutions.clear();
        record_best_fitnesses.clear();
    };

    vector<Real>        record_best_fitnesses;
    vector<int>         record_generations;
    vector<vector<Real>>    record_solutions;
};

#endif
