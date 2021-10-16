#!/bin/bash


# matde_problem, manytask10, cec50, mto_benchmark, Arm
problem_set="Arm"
experiment="${problemset}/DE/mto"
results_dir="Results/${experiment}"
mkdir -p $results_dir

tags=(zero)
total_tasks=(10)
tasks_start=1
runs=10

localrun=1 # 1 run on local machine, 0 run on remote cluster
executable='./AEMTO' # executable, e.g. ./AEMTo, MATDE, SBO, etc.

for tag in ${tags[@]}
do
    for tasks in ${total_tasks[@]}
    do     
        results_subdir="${tag}_${tasks}"
        cmd="$executable \
              -problem_set ${problem_set} \
              -total_tasks 1-$tasks \
              -total_runs $runs \
              -problem_name 1 \
              -results_dir $results_dir \
              -results_subdir ${results_subdir}"
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
