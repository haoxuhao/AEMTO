#ifndef __H_SET_PARAMS_H__
#define __H_SET_PARAMS_H__

#include <string>
#include "config.h"
#include "util.h"
#include "EA.h"


///https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c
class CmdParser{
    public:
        CmdParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }
        /// @author iain
        const std::string getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }
        string getCmdOption(const std::string &opt, string default_value) {
            string ret = getCmdOption(opt);
            if (ret == "") return default_value;
            return ret;
        }
        int getCmdOption(const std::string &opt, int default_value) {
            string ret = getCmdOption(opt);
            if (ret == "") return default_value;
            return stoi(ret);
        }
        double getCmdOption(const std::string &opt, double default_value) {
            string ret = getCmdOption(opt);
            if (ret == "") return default_value;
            return stod(ret);
        }
        /// @author iain
        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};

int SetParameters(IslandInfo &island_info, ProblemInfo &problem_info, \
    EAInfo &EA_info, Args &args, int argc, char** argv)
{
    CmdParser cmd_parser(argc, argv);
    args.results_dir = cmd_parser.getCmdOption("-results_dir", "Results/" + time_now());
    args.results_subdir = cmd_parser.getCmdOption("-results_subdir", "details/");
    args.problem_set = cmd_parser.getCmdOption("-problem_set", "Arm");
    args.problem_name = cmd_parser.getCmdOption("-problem_name", "1");
    args.UDim = cmd_parser.getCmdOption("-UDim", 50);
    args.popsize = cmd_parser.getCmdOption("-popsize", 100);
    args.record_interval = cmd_parser.getCmdOption("-record_interval", 100);
    args.total_runs = cmd_parser.getCmdOption("-total_runs", 10);
    args.G_max = cmd_parser.getCmdOption("-Gmax", 100);
    args.MTO = !cmd_parser.cmdOptionExists("--STO");
    string total_tasks_str = cmd_parser.getCmdOption("-total_tasks", "1-10");
    if(total_tasks_str.find("-") != total_tasks_str.npos) {
        vector<string> tmp = split(total_tasks_str, '-');
        for(int i = stoi(tmp[0]); i < stoi(tmp[1]) + 1; i++) {
            args.total_tasks.push_back(i);
        }
    }
    else if (total_tasks_str.find(",") != total_tasks_str.npos) {
        vector<string> tmp = split(total_tasks_str, ',');
        for(int i = 0; i < tmp.size(); i++) {
            args.total_tasks.push_back(stoi(tmp[i]));
        }
    }
    else {
        fprintf(stderr, "Error: invalid total tasks define: %s\n", total_tasks_str.c_str());
        abort();
    }
    
    EA_info.CR = 0.9;
    EA_info.F = 0.5;
    EA_info.UKTCR = 0.9;
    EA_info.LKTCR = 0.1;
    EA_info.STO = "DE";
    EA_info.ga_param.mu =  15;
    EA_info.ga_param.mum = 15;
    EA_info.ga_param.probswap = 0.5;

    island_info.island_size = args.popsize; /// TODO remove latter!

    cout << "===========================\n"
         << "Cmd Params: \n"
         << "problem set " << args.problem_set << "\n" 
         << "problem name " << args.problem_name << "\n"
         << "total tasks " << total_tasks_str << "\n"
         << "unified search space dims " << args.UDim << "\n"
         << "total runs " << args.total_runs << "\n"
         << "pop size " << args.popsize << "\n"
         << "Gmax " << args.popsize << "\n"
         << "record interval " << args.record_interval << "\n"
         << "results dir " << args.results_dir << "\n"
         << "results sub dir " << args.results_subdir << "\n"
         << "===========================" << endl;

    return 0;
}

int GetProblemInfo(const Args &args, ProblemInfo &base_info)
{
    int task_id = base_info.task_id;
    base_info.problem_def = args.problem_set;
    base_info.dim = args.UDim;
    base_info.min_bound = 0;
    base_info.max_bound = 1;
    string data_root = "./data";
    
    if(args.problem_set == "matde_problem")
    {
        base_info.shift_data_file = data_root + "/matde_problem/shift_task" 
            + to_string(task_id) + ".txt";
        static vector<string> func_names = {
            "sphere", "sphere", "sphere", "weierstrass",
             "rosenbrock", "ackley", "weierstrass", "schwefel", 
             "griewank", "rastrigin"
        };
        static vector<int> calc_dims {50, 50, 50, 25, 50, 50, 50, 50, 50, 50};
        base_info.task_id = task_id;
        base_info.calc_dim = calc_dims[task_id - 1];
        base_info.benchfunc_name = func_names[task_id - 1];
    }
    else if(args.problem_set == "manytask10")
    {
        string shift_tag = args.problem_name;
        base_info.shift_data_file = data_root + "/manytask10/" + 
            shift_tag + "/shift_task" + to_string(task_id) + ".txt";
        static vector<string> func_names = {
            "rosenbrock", "ackley", "schwefel", "griewank",
             "rastrigin", "rosenbrock", "ackley", "schwefel", 
             "griewank", "rastrigin"
        };
        static vector<int> calc_dims {50, 50, 50, 50, 50, 50, 50, 50, 50, 50};
        base_info.task_id = task_id;
        base_info.calc_dim = calc_dims[task_id - 1];
        base_info.benchfunc_name = func_names[task_id - 1];
    }
    else if(args.problem_set == "cec50")
    {
        base_info.task_id = task_id;
        int problem_id = stoi(args.problem_name);
        static vector<string> func_names = {
            "rosenbrock", "ackley", "rastrigin", 
            "griewank", "weierstrass", "schwefel"
        };
        base_info.benchfunc_name = func_names[problem_id - 1];
        base_info.calc_dim = 50;
        base_info.shift_data_file = data_root
            + "/CEC50/problem" + to_string(problem_id) 
            + "/shift_" + to_string(task_id) + ".txt";
        base_info.rotation_data_file = data_root
            + "/CEC50/problem" + to_string(problem_id) 
            + "/rotation_" + to_string(task_id) + ".txt";
    }
    else if(args.problem_set == "mto_benchmark")
    {
        assert(task_id <= 2 && "only two component tasks in this problem set");
        base_info.task_id = task_id;
        int problem_id = stoi(args.problem_name);
        const string problem_name = "problem" + to_string(problem_id);
        static unordered_map<string, vector<string>> problem_func_infos {
            {"problem1", {"griewank", "rastrigin"}},
            {"problem2", {"ackley", "rastrigin"}},
            {"problem3", {"ackley", "schwefel"}},
            {"problem4", {"rastrigin", "sphere"}},
            {"problem5", {"ackley", "rosenbrock"}},
            {"problem6", {"ackley", "weierstrass"}},
            {"problem7", {"rosenbrock", "rastrigin"}},
            {"problem8", {"griewank", "weierstrass"}},
            {"problem9", {"rastrigin", "schwefel"}}
        };
        static unordered_map<string, vector<int>> problem_dim_infos {
            {"problem1", {50, 50}},
            {"problem2", {50, 50}},
            {"problem3", {50, 50}},
            {"problem4", {50, 50}},
            {"problem5", {50, 50}},
            {"problem6", {50, 25}},
            {"problem7", {50, 50}},
            {"problem8", {50, 50}},
            {"problem9", {50, 50}},
        };
        base_info.benchfunc_name = problem_func_infos[problem_name][task_id - 1];
        base_info.calc_dim = problem_dim_infos[problem_name][task_id - 1];
        base_info.shift_data_file = data_root + "/mto_benchmark/" 
            + problem_name + "/shift_" + to_string(task_id) + ".txt";
        base_info.rotation_data_file = data_root + "/mto_benchmark/" 
            + problem_name + "/rotation_" + to_string(task_id) + ".txt";
    }
    else if(args.problem_set == "Arm")
    {
        base_info.dim = 50; 
        base_info.calc_dim = base_info.dim;
        base_info.arm_data_file = data_root + "/Arm/centroids_2000_2.dat";
    }
    else
    {
        fprintf(stderr, "Error: invalid problem definition or"
                        " the given problem is not implemented yet. \n");
        return -1;
    }
    return 0;
}

#endif
