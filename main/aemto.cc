#include <sstream>
#include <algorithm>
#include "config.h"
#include "communicator.h"
#include "SimpleIni.h"
#include "task.h"
#include "util.h"
#include "set_parameters.h"

Population select_to_import(vector<shared_ptr<Task>> &manytasks, int task_id)
{
    Population immigrants;
    auto task = manytasks[task_id];
    if (task->island_info.island_num == 1)
    {
        return immigrants;
    }
    unordered_map<int, int> selections;
    if (task->random_.RandRealUnif(0, 1) <= task->island_info.ada_import_epsilon)
    {
        selections = task->comm.selection.get_import_selections(true);
    }
    else
    {
        selections = task->comm.selection.get_import_selections();
    }
    for (auto sel : selections)
    {
        vector<real> fitnesses;
        const Population &pop_other = manytasks[sel.first]->GetPop();

        for(const auto & e : pop_other)
        {
            fitnesses.push_back(e.fitness_value);
        }
        vector<int> roulette_indices = task->random_.roulette_sample(fitnesses, sel.second);
        for (const auto &i : roulette_indices)
        {
            Individual ind = pop_other.at(i);
            ind.skill_factor = sel.first;
            immigrants.emplace_back(ind);
        }
    }
    return immigrants;
}

void execute_manytasks(vector<shared_ptr<Task>> &manytasks, int total_runs)
{
    long current_FEs = 0;
    int random_seed_base = 10000;
    int M = manytasks.size();
    long max_gens = manytasks[0]->max_gens;
    double task_start, g_start, comm_start;

    vector<real> total_time_cost_each_task;
    vector<real> total_transfer_time_cost_each_task;
    vector<real> last_update_rate_of_self_evolve;
    vector<real> last_update_rate_of_reuse;
    vector<int>  export_times_each_task(M, 0);
    vector<real> r_update_each_task(M, 0);
    vector<int> best_update_each_task(M, 0);

    for (int run_id = 1; run_id <= total_runs; run_id++)
    {
        fprintf(stderr, "\n==============================\n");
        fprintf(stderr, "===         run %d         ===\n", run_id);
        fprintf(stderr, "==============================\n\n");
        g_start = get_wall_time();
        srand(run_id * random_seed_base);
        fprintf(stderr, "Tasks initializing...\n");
        for (auto t : manytasks)
        {
            t->TaskInitialize(run_id);
            // printf("task %d init best fitness %f \n", t->problem_info.task_id, t->GetBestFitness());
        }
        auto task = manytasks[0];
        {
            real mean_fitness(0);
            for (const auto &t : manytasks)
            {
                mean_fitness += t->GetBestFitness();
            }
            mean_fitness /= manytasks.size();
            long fes = 1 * manytasks.size() * task->population.size();
            task->record.record_mean_fitness.push_back(make_pair(fes, mean_fitness));
            fprintf(stderr, "################# Run %d gen %d mean fitness %f\n", run_id, 0, mean_fitness);
        }
        fprintf(stderr, "Tasks initialized.\n");
        total_time_cost_each_task.assign(M, 0);
        total_transfer_time_cost_each_task.assign(M, 0);
        last_update_rate_of_self_evolve.assign(M, 0.0);
        last_update_rate_of_reuse.assign(M, 0.0);
        best_update_each_task.assign(M, 0);
        r_update_each_task.assign(M, 0);

        for (int gen = 1; gen < max_gens; gen++)
        {
            real gen_start = get_wall_time();
            for (int task_id = 0; task_id < manytasks.size(); task_id++)
            {
                task_start = get_wall_time();
                auto task = manytasks[task_id];
                if (! task->convergened())
                {
                    if (task->migrate.ImportCriteria(gen))
                    {
                        comm_start = get_wall_time();
                        // Population immigrants = task->comm.select_to_import();
                        Population immigrants = select_to_import(manytasks, task_id);
                        if (immigrants.size() > 0)
                        {
                            real best_fitness_before_reuse = task->GetBestFitness();
                            unordered_map<int, int> ns;
                            last_update_rate_of_reuse[task_id] = task->Reuse(immigrants, ns);
                            real best_fitness_after_reuse = task->GetBestFitness();
                            if (best_fitness_after_reuse < best_fitness_before_reuse){
                                best_update_each_task[task_id]++;
                                task->migrate.best_times_record_other++;
                            }
                            r_update_each_task[task_id] += last_update_rate_of_reuse[task_id];
                            task->migrate.add_update_rate_reuse(last_update_rate_of_reuse[task_id], gen);
                            task->comm.selection.update_import_pdf(ns, gen);
                            task->migrate.migration_counter++;
                            total_transfer_time_cost_each_task[task_id] += get_wall_time() - comm_start;

                            task->migrate.times_record_other += 100;
                        }
                    }
                    else
                    {
                        real bestf_before = task->GetBestFitness();
                        // evolution
                        last_update_rate_of_self_evolve[task_id] = task->EA_solver->Run(task->GetPop());
                        task->update_bestf();
                        r_update_each_task[task_id] += last_update_rate_of_self_evolve[task_id];
                        task->migrate.add_update_rate_self(last_update_rate_of_self_evolve[task_id], gen);
                        if (task->GetBestFitness() < bestf_before){
                            best_update_each_task[task_id]++;
                            task->migrate.best_times_record_self++;
                        }
                        task->migrate.times_record_self += 100;
                    }
                }
                if(task->island_info.record_details)
                {
                    task->record.record_import_prob.push_back(task->migrate.GetImportProb());
                    task->record.success_offsprings_rate.push_back(r_update_each_task[task_id] / (real)gen);
                    task->record.success_best_update_rate.push_back(best_update_each_task[task_id] / (real)gen);
                }
#ifdef DEEPINSIGHT
                printf("island %d self and reuse: [%.3f, %.3f, %.3f, %.12f]\n",
                            task->island_info.island_ID, 
                            task->migrate.get_Q_update_rate_self(),
                            task->migrate.get_Q_update_rate_reuse(),
                            task->migrate.GetImportProb(),
                            task->GetBestFitness());
#endif
                if (task->island_info.ada_import_prob == 1) {
                    task->migrate.UpdateImportProb();
                }
                
                total_time_cost_each_task[task_id] += get_wall_time() - task_start;
                if (gen % task->record.RECORD_INTERVAL == 0 || gen == max_gens || gen == 1)
                {
                    if (task->island_info.record_details)
                    {
                        task->record.recored_selection_probs.push_back(task->comm.selection.get_import_pdf());
                    }
                    Individual ind = task->find_best_individual();
                    fprintf(stderr, "task %d; run_id %d; gen: %d; migration count: %d; best fitness value: %.15f\n",
                            task->problem_info.task_id, run_id, gen, task->migrate.migration_counter, ind.fitness_value);

                    RecordInfo info;
                    info.best_fitness = ind.fitness_value;
                    info.generation = gen;
                    // info.elements = ind.elements;
                    task->record.time_section["total_time"] = total_time_cost_each_task[task_id];
                    task->record.time_section["total_kt_time"] =  total_transfer_time_cost_each_task[task_id];
                    task->record.RecordInfos(info);
                }
                //record the results
                if (gen == max_gens-1)
                {
                    task->record.FlushInfos();
                    Individual ind = task->find_best_individual();
                    real best_fitness = ind.fitness_value;
                    stringstream ss;
                    ss << "final solution: ";
                    for (auto e : ind.elements){ ss << e << ", "; }
                    ss << "fitness " << ind.fitness_value;
                    fprintf(stderr, "task %d, %s\n", task->problem_info.task_id, ss.str().c_str());
                    cerr << "record time is " << total_time_cost_each_task[task_id] << endl;
                }
            }
            
            auto task = manytasks[0];
            if (gen % task->record.RECORD_INTERVAL == 0 || gen == 1 || gen == (max_gens - 2))
            {
                real mean_fitness(0);
                for (const auto &t : manytasks)
                {
                    mean_fitness += t->GetBestFitness();
                }
                mean_fitness /= manytasks.size();
                long fes = (gen + 1) * manytasks.size() * task->population.size();
                task->record.record_mean_fitness.push_back(make_pair(fes, mean_fitness));
                fprintf(stderr, "################# Run %d gen %d mean fitness %f\n", run_id, gen, mean_fitness);
            }
            fprintf(stderr, "One gen cost time %.3f seconds\n", get_wall_time() - gen_start);
        }

        for (auto task : manytasks)
        {
            task->TaskUnInitialize();
        }
        fprintf(stderr, "One run cost time %.3f seconds\n", get_wall_time() - g_start);
    }
}

int main(int argc, char *argv[])
{
    clock_t start_time = get_wall_time();

    IslandInfo island_info;
    ProblemInfo problem_info;
    NodeInfo node_info;
    EAInfo EA_info;
    Args args;

    int ret = SetParameters(island_info, problem_info, node_info,
                            EA_info, args, argc, argv);
    if (ret != 0)
    {
        fprintf(stderr, "Error: set parameters error.\n");
        exit(-1);
    }
    node_info.node_num = 1;
    node_info.node_ID = 0;
    node_info.GPU_ID = args.gpu_id;
    node_info.GPU_num = args.nodes * args.gpus_per_node;
    node_info.nodes = args.nodes;


    island_info.island_num = args.total_tasks.size();
    island_info.task_ids.assign(args.total_tasks.begin(), args.total_tasks.end());

    // create tasks
    vector<shared_ptr<Task>> manytasks;
    CSimpleIni cfgs;

    cfgs.LoadFile(args.tasks_def.c_str());
    for (int i = 0; i < args.total_tasks.size(); i++)
    {
        island_info.island_ID = i;
        problem_info.task_id = args.total_tasks[i];
        problem_info.total_runs = args.total_runs;
        island_info.results_dir = args.results_dir;
        island_info.results_subdir = args.results_subdir;

        if (GetProblemInfo(cfgs, problem_info, args.use_unified_space) != 0)
        {
            fprintf(stderr, "Error: get task info error from file %s.\n", \
                args.tasks_def.c_str());
            exit(-1);
        }
        //add one task
        shared_ptr<Task> task = shared_ptr<Task>(std::move(
            new Task(node_info, problem_info, island_info, EA_info)));
        manytasks.push_back(task);
    }

    //spdlogger settings
    string default_log_dir = args.log_dir + "_aemto";
    mkdirs(default_log_dir.c_str());
    string default_log_file = default_log_dir + "/" + problem_info.problem_def + ".txt";

    // results dir
    mkdirs((args.results_dir + "/" + args.results_subdir).c_str());

    // execute tasks
    execute_manytasks(manytasks, args.total_runs);

    if(node_info.node_ID == 0)
    {
        fprintf(stderr, "Total elapsed time: %.3f s\n", get_wall_time() - start_time);
        ofstream ofs(args.results_dir + "/" + args.results_subdir + "/time.txt");
        ofs << ("Total elapsed time: " + to_string(get_wall_time() - start_time) + " s") << endl;
    }
    return 0;
}
