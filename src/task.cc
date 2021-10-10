#include <queue>
#include "task.h"
#include "util.h"
#include "comm.h"

Task::Task(
    NodeInfo &node_info, 
    ProblemInfo &problem_info, 
    IslandInfo &island_info, 
    EAInfo &EA_info) : 
        node_info(node_info), migrate(node_info), 
        record(node_info), comm(node_info),
        knowledge_reuse(island_info, problem_info, EA_info)
{
    this->problem_info = problem_info;
    this->island_info = island_info;
    this->EA_info = EA_info;
    max_gens = island_info.run_param.FEs / island_info.island_size + 1;

    if (EA_info.STO == "DE")
    {
        EA_solver.reset(std::move(new DE_CPU(node_info)));
    } else if (EA_info.STO == "GA")
    {
        EA_solver.reset(std::move(new GA_CPU())); 
    } else {
        throw invalid_argument("Unknown EA solver: ");
    }
    if (problem_info.problem_def == "Arm")
    {
        func = new ArmEvaluator();
    }else{
        func = new BenchFuncEvaluator();
    }
    func->Initialize(problem_info);
}

Task::~Task()
{
    delete func;
}
int Task::TaskInitialize(int run_id)
{
    problem_info.seed = run_id * 10000 * (problem_info.task_id);
    problem_info.run_ID = run_id;

    EA_solver->set_func(func);
    EA_solver->Initialize(island_info, problem_info, EA_info);
    EA_solver->InitializePopulation(population);
    knowledge_reuse.set_func_eval(func);
    migrate.Initialize(island_info, problem_info, EA_info);
    record.Initialize(island_info, problem_info, EA_info);
    comm.Initialize(island_info, problem_info);
    comm.update_shared_buffer(population);
    update_bestf();

    time_record = TimeSec();
    best_update_times = 0;
    individuals_update = 0;
    record_gens = 0;
    
    return 0;
}

int Task::TaskUnInitialize()
{
    EA_solver->Uninitialize();
    migrate.Uninitialize();
    record.Uninitialize();
    population.clear();
    comm.Uninitialize();
    return 0;
}

real Task::Reuse(Population &other, unordered_map<int, int> &ns)
{
    real update_rate = knowledge_reuse.Reuse(population, other, ns);
    update_bestf();
    return update_rate;
}

real Task::EA_solve()
{
    real r = EA_solver->Run(population);
    update_bestf();
    return r;
}
