#ifndef __H_SET_PARAMS_H__
#define __H_SET_PARAMS_H__

#include "config.h"
#include "util.h"
#include "EA.h"
#include "SimpleIni.h"

struct Args
{
    vector<int> total_tasks;
    int total_runs;
    int nodes;
    int gpus_per_node;
    uint gpu_id;
    string tasks_def;
    bool use_unified_space;
    bool debug;
    string params_file;
    string log_dir;
    string results_dir;
    string results_subdir;
    int kn_provid_thread_num;
    int cwu_thread; 
};


char* getParam(const char * needle, char* haystack[], int count)
{
    int i = 0;
    for (i = 0; i < count; i++) {
        if (strcmp(needle, haystack[i]) == 0) {
            if (i < count - 1) {
                return haystack[i + 1];
            }
        }
    }
    return 0;
}

int SetParameters(IslandInfo &island_info, ProblemInfo &problem_info, \
    NodeInfo &node_info, EAInfo &EA_info, Args &args, int argc, char** argv)
{
    char *p = NULL;
    p = getParam("-debug", argv, argc);
    args.debug = p ? true : false;

    p = getParam("-log_dir", argv, argc);
    args.log_dir = p? p : "tmp/emto_log";

    p = getParam("-results_dir", argv, argc);
    args.results_dir = p ? p : "Results/"+time_now();

    p = getParam("-results_subdir", argv, argc);
    args.results_subdir = p ? p : "details";

    p = getParam("-knp_thread_num", argv, argc);
    args.kn_provid_thread_num = p ? atoi(p) : 1;
    
    p = getParam("-cwu_thread", argv, argc);
    args.cwu_thread = p ? atoi(p) : 1;

    if (getParam("-total_tasks", argv, argc))
    {
        string str = getParam("-total_tasks", argv, argc);
        if(str.find("-") != str.npos)
        {
            vector<string> tmp;
            split(str, '-', tmp);
            
            for(int i=atoi(tmp[0].c_str()); i < atoi(tmp[1].c_str())+1; i++)
            {
                args.total_tasks.push_back(i);
            }
        }
        else if (str.find(",")!=str.npos)
        {
            vector<string> tmp;
            split(str, ',', tmp);
            for(int i = 0; i < tmp.size(); i++)
            {
                const char *tmp_function1 = tmp[i].c_str();
                args.total_tasks.push_back(atoi(tmp_function1));
            }
        }
        else{
            fprintf(stderr, "Error: get tasks error, invalid arguments: %s\n", str.c_str());
            return -1;
        }
    }

    p = getParam("-params_file", argv, argc);
    if (p)
    {
        args.params_file = p;
    }else{
        fprintf(stderr, "Error: get pamrams file error\n");
        return -1;
    }
    
    p = getParam("-tasks_def", argv, argc);
    if (p)
    {
        args.tasks_def = p;
    }else{
        fprintf(stderr, "Error: get tasks define file error\n");
        return -1;
    }
    
    p = getParam("-total_runs", argv, argc);
    args.total_runs = p ? atoi(p) : 1;
    p = getParam("-max_base_FEs", argv, argc);
    problem_info.max_base_FEs = p ? atoi(p) : 2000;
    p = getParam("-computing_time", argv, argc);
    problem_info.computing_time =  p ? atoi(p) : 500;

    p = getParam("-nodes", argv, argc);
    args.nodes = p ? atoi(p) : 1;
    p = getParam("-gpus_per_node", argv, argc);
    args.gpus_per_node = p ? atoi(p) : 1;
    p = getParam("-gpu_id", argv, argc);
    args.gpu_id = p ? atoi(p) : 0;

    p = getParam("-island_size", argv, argc);
    island_info.island_size = p ? atoi(p) : 100;
    p = getParam("-interval", argv, argc);
    island_info.interval = p ? atoi(p) : 10;
    p = getParam("-connection_rate", argv, argc);
    island_info.connection_rate = p ? atof(p) : 1;
    p = getParam("-export_rate", argv, argc);
    island_info.export_rate = p ? atof(p) : 0.01;
    p = getParam("-import_rate", argv, argc);
    island_info.import_rate = p ? atof(p) : 1;
    p = getParam("-buffer_manage", argv, argc);
    island_info.buffer_manage = p ? p : "random"; 
    p = getParam("-buffer_capacity", argv, argc);
    island_info.buffer_capacity = p ? atof(p) : 1;
 
    /* read common parameters from cfg file*/
    CSimpleIni MaTEA_cfgs;
    MaTEA_cfgs.LoadFile(args.params_file.c_str());
    
    EA_info.CR = MaTEA_cfgs.GetDoubleValue("DE", "CR", 0.8);
    EA_info.F = MaTEA_cfgs.GetDoubleValue("DE", "F", 0.5);
    EA_info.strategy_ID = MaTEA_cfgs.GetLongValue("DE", "strategy", 4);
    EA_info.LCR = MaTEA_cfgs.GetDoubleValue("DE", "LCR", 0.1);
    EA_info.UCR = MaTEA_cfgs.GetDoubleValue("DE", "UCR", 0.9);
    EA_info.UF = MaTEA_cfgs.GetDoubleValue("DE", "UF", 2);
    EA_info.LF = MaTEA_cfgs.GetDoubleValue("DE", "LF", 0.1);
    
    problem_info.dim = MaTEA_cfgs.GetLongValue("MaTDE", "U_DIM", 50);
    EA_info.ktc_cr = MaTEA_cfgs.GetDoubleValue("MaTDE", "ktc_cr", 1.0);
    EA_info.transfer_cross_over_rate = MaTEA_cfgs.GetDoubleValue("MaTDE", "transfer_cross_over_rate", 1);
    args.use_unified_space = MaTEA_cfgs.GetBoolValue("MaTDE", "use_unified_search_space", false);
    EA_info.UKTCR = MaTEA_cfgs.GetDoubleValue("MaTDE", "UKTCR", 0.9);
    EA_info.LKTCR = MaTEA_cfgs.GetDoubleValue("MaTDE", "LKTCR", 0.1);
    
    if(args.use_unified_space)
    {
        problem_info.max_bound = MaTEA_cfgs.GetDoubleValue("MaTDE", "U_UP_BOUND", 1);
        problem_info.min_bound = MaTEA_cfgs.GetDoubleValue("MaTDE", "U_LOW_BOUND", 0);
    }else{
        fprintf(stderr, "use original search space. \n");
    }

    string kttype = MaTEA_cfgs.GetValue("KT", "KTType", "UNI_CROSSOVER");
    if(kttype == "UNI_CROSSOVER"){
        island_info.transfer_type = UNI_CROSSOVER;
    }
    else if(kttype == "INSERT")
    {
        island_info.transfer_type = INSERT;
    }
    else if(kttype == "DE_BASE_VEC")
    {
        island_info.transfer_type = DE_BASE_VEC;
    }else{
        fprintf(stderr, "Invalid knowledge transfer type given %s. \n", kttype.c_str());
        return -1;
    }
    
    island_info.ada_param.alpha = MaTEA_cfgs.GetDoubleValue("AdaParam", "alpha", 0.5);
    island_info.ada_param.beta = MaTEA_cfgs.GetDoubleValue("AdaParam", "beta", 0.2);
    island_info.ada_param.Delta_T = MaTEA_cfgs.GetDoubleValue("AdaParam", "Delta_T", 100000);
    island_info.ada_param.Delta_T = MaTEA_cfgs.GetDoubleValue("AdaParam", "Delta_T", 100000);
    island_info.ada_param.pbase = MaTEA_cfgs.GetDoubleValue("AdaParam", "pbase", 0.5);
    island_info.ada_param.sample_batch = MaTEA_cfgs.GetLongValue("AdaParam", "sample_batch", 1000000);

    island_info.pmto_param.sync_gen = MaTEA_cfgs.GetBoolValue("ParallelParam", "sync_gen", false); 
    island_info.pmto_param.wait_pop = MaTEA_cfgs.GetBoolValue("ParallelParam", "wait_pop", false); 
    island_info.pmto_param.use_buffer = MaTEA_cfgs.GetBoolValue("ParallelParam", "use_buffer", false); 
    island_info.pmto_param.sync_gens = MaTEA_cfgs.GetLongValue("ParallelParam", "sync_gens", 100);
    island_info.pmto_param.cuda_EA_delta = MaTEA_cfgs.GetLongValue("ParallelParam", "cuda_EA_delta", 50);
     
    island_info.import_strategy = MaTEA_cfgs.GetValue("KT", "import_strategy", "Epsilon_ADA");
    island_info.ada_import_prob = MaTEA_cfgs.GetLongValue("KT", "ada_import_prob", 1);
    island_info.upper_import_prob = MaTEA_cfgs.GetDoubleValue("KT", "upper_import_prob", 0.9);
    island_info.lower_import_prob = MaTEA_cfgs.GetDoubleValue("KT", "lower_import_prob", 0.1);
    island_info.select_num = MaTEA_cfgs.GetLongValue("KT", "select_num", args.total_tasks.size() - 1);
    island_info.ada_import_epsilon = MaTEA_cfgs.GetDoubleValue("KT", "ada_import_epsilon", 0.0);
    island_info.export_strategy = MaTEA_cfgs.GetValue("KT", "export_strategy", "EDT");
    island_info.reward_self = MaTEA_cfgs.GetDoubleValue("KT", "reward_self", 1.0);
    island_info.ada_import_strategy = MaTEA_cfgs.GetValue("KT", "ada_import_strategy", "Matching");

    island_info.emmigration_strategy = MaTEA_cfgs.GetValue("KT", "emmigration_strategy", "best");
    island_info.buffer_sampling = MaTEA_cfgs.GetValue("global", "buffer_sampling", "best");
    island_info.buffer_manage = MaTEA_cfgs.GetValue("global", "buffer_manage", "first");

    island_info.import_interval = MaTEA_cfgs.GetDoubleValue("KT", "import_interval", 10);
    island_info.import_prob = MaTEA_cfgs.GetDoubleValue("KT", "import_prob", 0.1);
    island_info.export_interval = MaTEA_cfgs.GetDoubleValue("KT", "export_interval", 10);
    island_info.export_prob = MaTEA_cfgs.GetDoubleValue("KT", "export_prob", 1.0);
    island_info.import_rate = MaTEA_cfgs.GetDoubleValue("KT", "import_rate", 0.2);
    island_info.export_rate = MaTEA_cfgs.GetDoubleValue("KT", "export_rate", 1.0);
    island_info.buffer_capacity = MaTEA_cfgs.GetDoubleValue("KT", "buffer_capacity", 1);
    island_info.island_size = MaTEA_cfgs.GetLongValue("global", "population", 100);
    island_info.run_param.FEs = MaTEA_cfgs.GetLongValue("global", "FEs", 100000);
    problem_info.max_base_FEs = MaTEA_cfgs.GetLongValue("global", "base_FEs", 2000);
    problem_info.computing_time = MaTEA_cfgs.GetLongValue("global", "computing_time", 1000);
    args.total_runs = MaTEA_cfgs.GetLongValue("global", "total_runs", 1);
    island_info.print_interval = MaTEA_cfgs.GetLongValue("global", "print_interval", 50); 
    island_info.record_details = MaTEA_cfgs.GetBoolValue("global", "record_details", 0);

    EA_info.group_size = island_info.island_size;
    EA_info.group_num = island_info.island_size / EA_info.group_size;
    EA_info.STO = MaTEA_cfgs.GetValue("EA", "STO", "DE");
    EA_info.ktcr_strategy = MaTEA_cfgs.GetValue("EA", "ktcr_strategy", "UNI");
    if(EA_info.STO == "GA")
    {
        EA_info.ga_param.mu = MaTEA_cfgs.GetDoubleValue("GA", "mu", 20);
        EA_info.ga_param.mum = MaTEA_cfgs.GetDoubleValue("GA", "mum", 15);
        EA_info.ga_param.probswap = MaTEA_cfgs.GetDoubleValue("GA", "probswap", 0.5);
        printf("set ga param mu %.2f mum %.2f probswap %.2f\n", EA_info.ga_param.mu, EA_info.ga_param.mum, EA_info.ga_param.probswap);
    }
    return 0;

}

int GetProblemInfo(CSimpleIni &cfgs, ProblemInfo &base_info, bool use_unified)
{
    //load global problem data
    string problem_def = cfgs.GetValue("global", "problem_def", "MaTDE10SOTasks");
    string bin_root = cfgs.GetValue("global", "bin_root", "/fred/oz121/hxu/EMTO_GPU/bin/");
    int task_id = base_info.task_id;
    base_info.problem_def = problem_def;
    base_info.bin_root = bin_root;
    if(problem_def == "MaTDE10SOTasks")
    {
        string data_dir = bin_root + string(cfgs.GetValue("global", "shift_data_dir", "null"));
        base_info.shift_data_root = data_dir;
       
        base_info.shift_data_prefix = cfgs.GetValue("global", "shift_data_file_prefix", "null");
        
        base_info.rotation_data_root = (bin_root + string(cfgs.GetValue("global", "rotation_data_dir", "null")));
        base_info.rotation_data_prefix = cfgs.GetValue("global", "rotation_data_file_prefix", "null");

        base_info.task_id = task_id;
        string task_name = "task" + std::to_string(task_id);
        base_info.o_max_bound = cfgs.GetDoubleValue(task_name.c_str(), "max_bound", 100);
        base_info.o_min_bound = cfgs.GetDoubleValue(task_name.c_str(), "min_bound", -100);
        base_info.calc_dim    = (int)cfgs.GetLongValue(task_name.c_str(), "dim", 50);
        base_info.func_id     = (int)cfgs.GetLongValue(task_name.c_str(), "function_id", 0);
        if(!use_unified)
        {
            base_info.max_bound = base_info.o_max_bound;
            base_info.min_bound = base_info.o_min_bound;
        }
    }
    else if(problem_def == "CEC50SOTasks")
    {
        base_info.task_id = task_id;
        int problem_id = cfgs.GetLongValue("global", "problem_id", 1);
        string problem_info_section = "problem"+std::to_string(problem_id);
        base_info.calc_dim = base_info.dim;
        base_info.o_max_bound = cfgs.GetDoubleValue(problem_info_section.c_str(), "max_bound", 100);
        base_info.o_min_bound = cfgs.GetDoubleValue(problem_info_section.c_str(), "min_bound", -100);
        base_info.func_id = (int)cfgs.GetLongValue(problem_info_section.c_str(), "function_id", 0);

        base_info.shift_data_root = (bin_root + cfgs.GetValue(problem_info_section.c_str(), "shift_data_dir", "null"));
        base_info.rotation_data_root = (bin_root + cfgs.GetValue(problem_info_section.c_str(), "rotation_data_dir", "null"));

        base_info.shift_data_prefix =  cfgs.GetValue(problem_info_section.c_str(), "shift_data_file_prefix", "null");
        base_info.rotation_data_prefix = cfgs.GetValue(problem_info_section.c_str(), "rotation_data_file_prefix", "null");

        if(!use_unified)
        {
            base_info.max_bound = base_info.o_max_bound;
            base_info.min_bound = base_info.o_min_bound;
        }
    }
    else if (problem_def == "ManMany")
    {
        base_info.task_id = task_id;
        int task_index = task_id - 1;
        base_info.dim = cfgs.GetLongValue("global", "dim", 10); 
        base_info.calc_dim = base_info.dim;
        int base_funcs = cfgs.GetLongValue("global", "basefuncs", 5);
        string func_id_str = ("func_id" + to_string(task_index % base_funcs));
        base_info.func_id = cfgs.GetLongValue("funcs", func_id_str.c_str(), 0);
        vector<string> range = split(cfgs.GetValue("ranges", func_id_str.c_str(), "-50,50"), ',');

        base_info.o_max_bound = std::stof(range[1]);
        base_info.o_min_bound = std::stof(range[0]); 
        
        base_info.shift_data_root = (bin_root + cfgs.GetValue("global", "shift_data_dir", "null"));
        base_info.rotation_data_root = (bin_root + cfgs.GetValue("global", "rotation_data_dir", "null"));

        base_info.shift_data_prefix =  cfgs.GetValue("global", "shift_data_file_prefix", "null");
        base_info.rotation_data_prefix = cfgs.GetValue("global", "rotation_data_file_prefix", "null");

        if(!use_unified)
        {
            base_info.max_bound = base_info.o_max_bound;
            base_info.min_bound = base_info.o_min_bound;
        }
    }
    else if(problem_def == "Arm")
    {
        base_info.o_max_bound = cfgs.GetDoubleValue("range", "max", 1);
        base_info.o_min_bound = cfgs.GetDoubleValue("range", "min", 0);
        base_info.max_bound = base_info.o_max_bound;
        base_info.min_bound = base_info.o_min_bound; 
        base_info.dim = cfgs.GetLongValue("global", "dim", 10); 
        base_info.calc_dim = base_info.dim;
        base_info.shift_data_root = (bin_root + cfgs.GetValue("global", "shift_data_dir", "null"));
        base_info.shift_data_prefix = cfgs.GetValue("global", "shift_data_file_prefix", "null");
    }
    else
    {
        fprintf(stderr, "Error: invalid problem definition or the given problem is not implemented yet. \n");
        return -1;
    }
    return 0;
}


#endif

