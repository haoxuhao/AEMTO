#include "evaluator.h"
#include "util.h"

int BenchFuncEvaluator::Initialize(const ProblemInfo &problem_info)
{
    problem_info_ = problem_info;
	calc_dim_ = problem_info_.calc_dim;
    func_id_ = problem_info.func_id;
    task_id_ = problem_info.task_id;
	dim_ = problem_info.dim;
	cf_index_ = 0;
	cf_func_num_ = 1;
	ye = new real[dim_];
	ze = new real[dim_];
	x_bound = new real[dim_];
	pop_original_ = new real[dim_];
	
	//scale from unified search space to original search space
    scale_rate_ = (problem_info_.o_max_bound-problem_info_.o_min_bound)/(problem_info_.max_bound-problem_info_.min_bound);
	fixed_shift_ = 0;
	
    if(func_id_>=23)
    {
        flag_composition_=true;
    }
    else
    {
        flag_composition_=false;
    }

    switch (func_id_)
	{
	case 23:
		cf_func_num_ = 5;
		break;
	case 24:
		cf_func_num_ = 3;
		break;
	case 25:
		cf_func_num_ = 3;
		break;
	case 26:
		cf_func_num_ = 5;
		break;
	case 27:
		cf_func_num_ = 5;
		break;
	case 28:
		cf_func_num_ = 5;
		break;
	case 29:
		cf_func_num_ = 3;
		break;
	case 30:
		cf_func_num_ = 3;
		break;
	default:
		break;
	}
	if (problem_info_.problem_def == "ManMany")
	{
		string shifts_file = problem_info_.shift_data_root + "/" + problem_info_.shift_data_prefix + ".dat";
		load_shifts_from_singlefile(shifts_file);
	} else {
		LoadData_matea();
	}
    return 0;
}

int BenchFuncEvaluator::Uninitialize()
{   
    CEC2014::Uninitialize();
    return 0;
}

int BenchFuncEvaluator::transfer_to_original_space(const vector<real> & elements)
{
	for(int i=0; i<dim_; i++)
	{
		pop_original_[i] = scale_rate_*elements[i] - scale_rate_*problem_info_.min_bound + problem_info_.o_min_bound;
	}
	return 0;
}
real BenchFuncEvaluator::EvaluateFitness(const vector<real> & elements, mutex &mtx)
{
	// real f;
	// mtx.lock();
	// f = EvaluateFitness(elements);
	// mtx.unlock();
	// return f;
}

real BenchFuncEvaluator::EvaluateFitness(const vector<real> & elements)
{
	real fitness_value = 0;
	// transfer_to_original_space(elements);
	real *pop_original = new real[dim_];
	for(int i = 0; i < dim_; i++)
	{
		pop_original[i] = scale_rate_*elements[i] - scale_rate_*problem_info_.min_bound + problem_info_.o_min_bound;
	}
    switch(func_id_)
    {
        case 0:
            sphere_func(pop_original, &fitness_value, calc_dim_, OShift, Mdata, 1, rotation_flag_, fixed_shift_, 1);
			break;
		case 4:
			rosenbrock_func(pop_original, &fitness_value, calc_dim_, OShift, Mdata, 1, rotation_flag_, fixed_shift_, 1);
			break;
		case 5:
			ackley_func(pop_original, &fitness_value, calc_dim_, OShift, Mdata, 1, rotation_flag_, fixed_shift_, 1);
			break;
		case 6:
			weierstrass_func(pop_original, &fitness_value, calc_dim_, OShift, Mdata, 1, rotation_flag_, fixed_shift_, 1);
			break;
		case 7:
			griewank_func(pop_original, &fitness_value, calc_dim_, OShift, Mdata, 1, rotation_flag_, fixed_shift_, 1);
			break;
		case 8:
			rastrigin_func(pop_original, &fitness_value, calc_dim_, OShift, Mdata, 1, rotation_flag_, fixed_shift_, 1);
			break;
		case 9:
			schwefel_func(pop_original, &fitness_value, calc_dim_, OShift, Mdata, 1, rotation_flag_, fixed_shift_, 1);
			break;
		default:
			fprintf(stderr, "Error: invalid function id: %d\n", func_id_);
			break;
    }
	delete[]pop_original;
	return fitness_value;
}

void BenchFuncEvaluator::load_shifts_from_singlefile(string file_name)
{
    OShift = new real[dim_ * cf_func_num_];
	Mdata = new real[dim_ * dim_ * cf_func_num_];
	int cnt = 0;
	string sLine = "";
    ifstream read;
    read.open(file_name);
	if (!read)
	{
		fprintf(stderr, "Error: can not open file: %s.\n", file_name.c_str());	
		exit(0);
	}
	while (cnt != problem_info_.task_id && getline(read, sLine)){ 
		++cnt;
	}
	if (cnt == problem_info_.task_id)
	{
		vector<string> shifts_str = split(sLine, ' ', false);
		assert(shifts_str.size() == dim_ && "dimension incomplete");
		int j = 0;
		for (const auto e: shifts_str)
		{
			OShift[j++] = stof(e);
		}
	} else {
		fprintf(stderr, "Error: get line %d from file %s.\n", problem_info_.task_id, file_name.c_str());	
		exit(0);
	}
	// using default identy matrix
	for (int k = 0; k < cf_func_num_; k++)
    {
        for (int i = 0; i < dim_; i++)
        {
            for (int j = 0; j < dim_; j++)
			{
				if (i == j)
					Mdata[j + i * dim_ + k * dim_ * dim_] = 1;
				else
					Mdata[j + i * dim_ + k * dim_ * dim_] = 0;
			}
        }	
    }
	fprintf(stderr, "note: read shift data from %s in line %d, identity rotation matrix is used.\n", file_name.c_str(), cnt);
}

int BenchFuncEvaluator::LoadData_matea()
{
	char fileName[256]={0};
    FILE *fpt;

    Mdata = new real[dim_ * dim_ * cf_func_num_];
    OShift = new real[dim_ * cf_func_num_];
    SS = new int[dim_ * cf_func_num_];

    //set the default shift and rotation data
	for (int i = 0; i < cf_func_num_; i++)
    {
        for (int j = 0; j < dim_; j++)
		{
			OShift[j + i * dim_] = 0;
			SS[j + i * dim_] = j;
		}
    }
		
	for (int k = 0; k < cf_func_num_; k++)
    {
        for (int i = 0; i < dim_; i++)
        {
            for (int j = 0; j < dim_; j++)
			{
				if (i == j)
					Mdata[j + i * dim_ + k * dim_ * dim_] = 1;
				else
					Mdata[j + i * dim_ + k * dim_ * dim_] = 0;
			}
        }	
    }
		
    /* Load Matrix Mdata*/
	sprintf(fileName, "%s/%s%d.txt", problem_info_.rotation_data_root.c_str(), problem_info_.rotation_data_prefix.c_str(), task_id_);
	fpt = fopen(fileName, "r");
	if (fpt == NULL)
	{
		fprintf(stderr, "warning: can not open file: %s; use identity matrix as default rotation matrix.\n", fileName);
		problem_info_.is_rotate = false;
		rotation_flag_ = 0;
	}else
    {
		fprintf(stderr, "task %d load rotation data from %s\n", problem_info_.task_id, fileName);
        problem_info_.is_rotate = true;
		rotation_flag_ = 1;
		for (int k = 0; k < cf_func_num_; k++)
		{
				for (int i = 0; i < dim_; i++)
				{
                    for (int j = 0; j < dim_; j++)
                    {
#ifdef DOUBLE_PRECISION
                        fscanf(fpt, "%lf", &Mdata[i + (j + k * dim_) * dim_]);
#else
                        fscanf(fpt, "%f", &Mdata[i + (j + k * dim_) * dim_]);
#endif
                    }
				}
		}
		fclose(fpt);
    }
	
	/* Load shift_data */
	sprintf(fileName, "%s/%s%d.txt", problem_info_.shift_data_root.c_str(), problem_info_.shift_data_prefix.c_str(), task_id_);
	fpt = fopen(fileName, "r");
	if (fpt == NULL)
	{
        fprintf(stderr, "warning: can not open file: %s; use 0 as default shift vector.\n", fileName);
	}
    else
    {
		fprintf(stderr, "task %d load shift data from %s\n", problem_info_.task_id, fileName);
        if (flag_composition_)
		{
			for (int i = 0; i < cf_func_num_; i++)
			{
				for (int j = 0; j < dim_; j++)
				{
#ifdef DOUBLE_PRECISION
                    fscanf(fpt, "%lf", &OShift[j + i * dim_]);
#else
                    fscanf(fpt, "%f", &OShift[j + i * dim_]);
#endif
				}
			}

		}
		else
		{
			for (int i = 0; i < dim_; i++)
			{

	#ifdef DOUBLE_PRECISION
				fscanf(fpt, "%lf", &OShift[i]);
	#else
				fscanf(fpt, "%f", &OShift[i]);
	#endif
			}
		}
		fclose(fpt);
    }
	/* Load Shuffle_data */

	if (func_id_ >= 17 && func_id_ <= 22)
	{
		sprintf(fileName, "%s/data/CEC2014-benchmark/shuffle_data_%d_D%d.txt", problem_info_.bin_root.c_str(), func_id_, dim_);
		fpt = fopen(fileName, "r");
		if (fpt == NULL)
		{
			printf("\n Error: Cannot open input file for reading \n");
		}
		SS = new int[dim_ * cf_func_num_];
		if (SS == NULL)
			printf("\nError: there is insufficient memory available!\n");
		for (int i = 0; i<dim_; i++)
		{
			fscanf(fpt, "%d", &SS[i]);
		}
		fclose(fpt);
	}
	else if (func_id_ == 29 || func_id_ == 30)
	{
		sprintf(fileName, "%s/data/CEC2014-benchmark/shuffle_data_%d_D%d.txt", problem_info_.bin_root.c_str(), func_id_, dim_);
		fpt = fopen(fileName, "r");
		if (fpt == NULL)
		{
			printf("\n Error: Cannot open input file for reading \n");
		}
		SS = new int[dim_ * cf_func_num_];
		if (SS == NULL)
			printf("\nError: there is insufficient memory available!\n");
		for (int i = 0; i<dim_*cf_func_num_; i++)
		{
			fscanf(fpt, "%d", &SS[i]);
		}
		fclose(fpt);
	}
    return 0;

}

int ArmEvaluator::load_data()
{
	string shifts_file = problem_info_.shift_data_root + "/" + problem_info_.shift_data_prefix + ".dat";
	int cnt = 0;
	string sLine = "";
    ifstream read;
    read.open(shifts_file);
	if (!read)
	{
		fprintf(stderr, "Error: can not open file: %s.\n", shifts_file.c_str());	
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
		fprintf(stderr, "Error: get line %d from file %s.\n", problem_info_.task_id, shifts_file.c_str());	
		exit(0);
	}
	fprintf(stderr, "Note load data from %s in line %d\n", shifts_file.c_str(), problem_info_.task_id);
}

int ArmEvaluator::Initialize(const ProblemInfo &problem_info)
{
	// first initialize
	if (task.size() == 0)
	{
		problem_info_ = problem_info;
		load_data();
		real length_norm = task[1] / (real)problem_info_.dim;
		lengths.resize(problem_info_.dim + 1, length_norm);
		lengths[0] = 0;
		angular_range = PI * 2 * task[0] / (real)problem_info_.dim;
		n_dofs = problem_info_.dim;
	}
	return 0;
}

#include "Eigen/Dense"

typedef double Real;

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
	// Mat mat = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
	// Mat b = {{0}, {0}, {0}, {1}};
	// for (int i = 0; i <= n_dofs; i++)
	// {
	// 	Mat m = { { cos(p[i]), -sin(p[i]), 0, lengths[i] },
	// 			{ sin(p[i]),  cos(p[i]), 0, 0 },
	// 			{ 0, 0, 1, 0 },
	// 			{ 0, 0, 0, 1 }};
	// 	mat = matrix_multiply(mat, m);
	// 	Mat v = matrix_multiply(mat, b);
	// 	joint_xy.push_back({v[0][0], v[1][0]});
	// }
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
