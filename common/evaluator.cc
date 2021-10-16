#include "evaluator.h"
#include "util.h"
#include "benchfunc.h"


BenchFuncEvaluator::BenchFuncEvaluator(const ProblemInfo &problem_info)
{
	problem_info_ = problem_info;
	//load shifts and rotation matrix
	LoadTaskData(); 

	//parameters for scaling x from [0, 1]^D to task specific space
	auto func_range = func_search_range.find(problem_info_.benchfunc_name);
	assert(func_range != func_search_range.end() && "unknown function name");
	Real lb = func_range->second[0];
	Real ub = func_range->second[1];
	Real uni_lb = 0;
	Real uni_ub = 1;
	scale_rate_ = (ub - lb) / (uni_ub - uni_lb);
	bias_vec_ = Eigen::VectorXd::Constant(
		problem_info_.calc_dim, -scale_rate_ * uni_lb + lb);
}

Real BenchFuncEvaluator::EvaluateFitness(const vector<Real> & elements)
{
	auto elements_to_eval = elements;
	Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
		elements_to_eval.data(), problem_info_.calc_dim);
	if (is_rotate_) {
		b = rM_ * (scale_rate_*b + bias_vec_) - shift_;
	} else {
		b = scale_rate_*b + bias_vec_ - shift_;
	}
	thread_local static const unordered_map<string, 
		std::function<Real(vector<Real>&)>> name_func_table {
        {"sphere", benchfunc::sphere}, 
        {"weierstrass", benchfunc::weierstrass},
        {"rosenbrock", benchfunc::rosenbrock}, 
        {"ackley",benchfunc::ackley}, 
        {"schwefel", benchfunc::schwefel},
        {"griewank", benchfunc::griewank}, 
        {"rastrigin", benchfunc::rastrigin}
    };
	vector<Real> processed_elements(b.data(), b.data() + b.size());

	return name_func_table.at(problem_info_.benchfunc_name)(processed_elements);
}

void BenchFuncEvaluator::LoadTaskData()
{
	fprintf(stderr, "task %d, shift data file %s, rotation data file %s\n",
		problem_info_.task_id, problem_info_.shift_data_file.c_str(), 
		problem_info_.rotation_data_file.c_str());
	string sLine = "";
    ifstream read;
    read.open(problem_info_.rotation_data_file);
	if (!read)
	{
		fprintf(stderr, "Error: can not open rotation data file: %s.\n", 
			problem_info_.rotation_data_file.c_str());
	} else {
		rM_.resize(problem_info_.calc_dim, problem_info_.calc_dim);
		for (int i = 0; i < problem_info_.calc_dim; i++) {
			getline(read, sLine);
			vector<string> shifts_str = split(sLine, ' ', false);
			assert(shifts_str.size() == problem_info_.calc_dim 
				   && "rotation matrix dim != problem dim");
			for (int j = 0; j < problem_info_.calc_dim; j++) {
				rM_(i, j) = std::stod(shifts_str[j]);
			}
		}
		is_rotate_ = true;
	}
	read.close();
	read.open(problem_info_.shift_data_file);
	if (!read) {
		fprintf(stderr, "Error: can not open file: %s.\n", problem_info_.shift_data_file.c_str());	
		abort();
	} else {
		shift_.resize(problem_info_.calc_dim);
		getline(read, sLine);
		vector<string> shifts_str = split(sLine, ' ', false);
		for (int j = 0; j < problem_info_.calc_dim; j++) {
			shift_(j) = std::stod(shifts_str[j]);
		}
	}
}

void ArmEvaluator::LoadTaskData()
{
	int cnt = 0;
	string sLine = "";
    ifstream read;
    read.open(problem_info_.arm_data_file);
	if (!read)
	{
		fprintf(stderr, "Error: can not open file: %s.\n", problem_info_.arm_data_file.c_str());	
		exit(0);
	}
	while (cnt != problem_info_.task_id && getline(read, sLine)){ 
		++cnt;
	}
	if (cnt == problem_info_.task_id)
	{
		vector<string> shifts_str = split(sLine, ' ', false);
		assert(shifts_str.size() == 2 && "features dim 2 required.");
		for (const auto e: shifts_str)
		{
			task.push_back(stof(e));
		}
	} else {
		fprintf(stderr, "Error: get line %d from file %s.\n", problem_info_.task_id, problem_info_.arm_data_file.c_str());	
		exit(0);
	}
	fprintf(stderr, "Note load data from %s in line %d\n", problem_info_.arm_data_file.c_str(), problem_info_.task_id);
}

ArmEvaluator::ArmEvaluator(const ProblemInfo &problem_info)
{
	problem_info_ = problem_info;
	LoadTaskData();
	Real length_norm = task[1] / (Real)problem_info_.dim;
	lengths.resize(problem_info_.dim + 1, length_norm);
	lengths[0] = 0;
	angular_range = PI * 2 * task[0] / (Real)problem_info_.dim;
	n_dofs = problem_info_.dim;
}

vector<Real> ArmEvaluator::fw_kinematics(vector<Real> & command)
{
	vector<Real> p = command;
	assert(p.size() == n_dofs);
	Mat joint_xy;
	
	Eigen::Matrix4d mat;
	mat << 1, 0, 0, 0,
		   0, 1, 0, 0,
		   0, 0, 1, 0,
		   0, 0, 0, 1;
	Eigen::Vector4d b(0, 0, 0, 1);
	for (int i = 0; i <= n_dofs; i++)
	{
		Eigen::Matrix4d m;
		m << cos(p[i]), -sin(p[i]), 0, lengths[i],
		     sin(p[i]),  cos(p[i]), 0, 0,
             0, 0, 1, 0,
			 0, 0, 0, 1;
	    mat = mat * m;
		Eigen::Vector4d v = mat * b;
		joint_xy.push_back({v[0], v[1]});
	}
	return joint_xy[joint_xy.size() - 1];
}

Real ArmEvaluator::EvaluateFitness(const vector<Real> & elements)
{
	vector<Real> command = elements;
	for (int i = 0; i < command.size(); i++)
	{
		command[i] = (command[i] - 0.5) * angular_range;
	}
	
	vector<Real> ef = fw_kinematics(command);
    // l2 lorm
	Real cost = 0;
	const Real target = 0.5;
	for (int i = 0; i < ef.size(); i++)
	{
		cost += pow((ef[i] - target), 2);
	}
	cost = sqrt(cost);
	return cost;
}
