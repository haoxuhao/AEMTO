#ifndef __H_KNOWLEDGE_TRANS_H__
#define __H_KNOWLEDGE_TRANS_H__

/**
 * Knoledge transfer branch
 *  - adaptive selection module, 
 *  - communication module,
 *  - reuse module
 * 
 * Main procedures:
 *  - Sample knowledge from other tasks. May trigger the 
 *    inter-process communication
 *  - Conduct reuse operation to generate the next offsprings
 *  - Concate the offspring and the current pop to generate the 
 *    intermediate pop
 *  - Do survival selection to generate the next population
 * 
 * Input: pop_{i, g}
 * Output: pop_{i, g+1}
 * Params: EA info, task info, island info, node info.
 * 
 * Note: the selection probabilities and the corresponding internal 
 * variables are updated after this procedure
 * */

#include "config.h"
#include "EA.h"
#include "evaluator.h"
#include "random.h"


class KnowledgeReuse
{
private:
    IslandInfo island_info_;
    EAInfo ea_info_;
    ProblemInfo problem_info_;

    Evaluator* func_eval_;

    Random random_;
public:
    KnowledgeReuse(
        IslandInfo &island_info, 
        ProblemInfo &problem_info, 
        EAInfo &ea_info) : island_info_(island_info),
                           problem_info_(problem_info),
                           ea_info_(ea_info){};
    ~KnowledgeReuse(){};
    void set_func_eval(Evaluator *func) {func_eval_ = func;};

    unordered_map<int, int> evolve(Population &pop);
    real Reuse(Population &pop, Population &other, unordered_map<int, int> &success_insert_table);
    void Initialize() {};
    void UnInitialize() {};
};

#endif