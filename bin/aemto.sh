#!/bin/bash

#SBATCH --ntasks=32
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1000
#SBATCH --time=11:40:00

echo "batch file args: $*"
if [ "$#" != "2" ] # submit through `sbatch ameto.sh` not by the run.sh 
then
    problemset="Arm"
    params_file="./cfgs/params.cfg"
    experiment="${problemset}/DE/mto"
    results_dir="Results/${experiment}"
    mkdir -p $results_dir
    cp $params_file $results_dir
    params_file="${results_dir}/$(basename $params_file)" # using new backup cfg
    echo "set backup params file ${params_file}"
else
    params_file=$1
    results_dir=$2
fi
echo "param file $params_file; results dir $results_dir"

# problem tags
# tags=(zero small median large)
# tags=("problem1" "problem2" "problem3" "problem4" "problem5" "problem6")
# tags=("problem1" "problem2" "problem3" "problem4" "problem5" "problem6" "problem7" "problem8" "problem9")
# tags=(zero small median large) 
tags=(zero)

# number of tasks for each problem, e.g. here test problem is zero_200
total_tasks=(10)

tasks_start=1

localrun=1 # 1 run on local machine, 0 run on remote cluster
executable='./AEMTO' # executable file, e.g. ./AEMTo, MATDE, SBO, etc.

for tag in ${tags[@]} 
do
    for tasks in ${total_tasks[@]}
    do     
        #tasks_def="./cfgs/MaTDE_10_tasks_def.cfg" # matde_problem
        #tasks_def="./cfgs/CEC_competition_50_SO_tasks_${tag}.cfg" # CEC50
        #tasks_def="./cfgs/ManyTask10/manytask_problem5_${tag}.cfg" # ManyTask10
        #tasks_def="./cfgs/mto_benchmark/multi_task_benchmark_${tag}.cfg" # mtobenchmark
        #tasks_def="./cfgs/ManMany/uni_shift_${tag}.cfg" # manmany
        tasks_def="./cfgs/ArmControl/arm_origin.cfg" # Arm
        
        results_subdir="${tag}_${tasks}"
        
        cmd="$executable -total_tasks 1-$tasks -tasks_def ${tasks_def} -params_file ${params_file} -nodes 1 -gpus_per_node 1 -results_dir $results_dir -results_subdir ${results_subdir}"
        echo "cmd: ${cmd}"
        if [ $localrun = "1" ]
        then
           $cmd
        else
           srun $cmd
        fi
    done  
done
wait

echo "concate results files..."
for tag in ${tags[@]}
do
    for job in ${total_tasks[@]}
    do
        echo "processing ${tag}_${job}"
        results_subdir="${tag}_${job}"
        results_compact="${results_dir}/${results_subdir}/compact.txt"
        rm -f $results_compact
        echo "start_end, ${tasks_start},${job}" >> $results_compact 
        for((i=${tasks_start}; i<=${job}; i ++))
        do 
            res_file=${results_dir}/${results_subdir}/res_task_$i.json
            cat $res_file >> $results_compact
            rm -f $res_file
        done
    done
done

>&2 echo "done. results dir: ${results_dir}"
