#!/bin/bash

# problems in different problem sets
# uncomment to run on a specific problem set

########### cec50 6 problems #################
# tags=(problem1 problem2 problem3 problem4 problem5 problem6) 
# tasks_start=1
# total_tasks=(50)
# problem_set="cec50"
##############################################

########### mtobenchmark 9 problems ##########
# tags=(problem1 problem2 problem3 problem4 problem5 problem6 problem7 problem8 problem9) 
# tasks_start=1
# total_tasks=(2)
# problem_set="mtobenchmark"
##############################################

############ manytask10 ######################
# tags=(zero small median large)
# tasks_start=1
# total_tasks=(10)
# problem_set="manytask10"
##############################################

############## matde problem #################
tags=(zero)
tasks_start=1
total_tasks=(10)
problem_set="matde_problem"
##############################################

############ Arm problem #####################
# tags=(zero)
# tasks_start=1
# total_tasks=(2000)
# problem_set="Arm"
##############################################


algo="AEMTO"
results_dir="Results/${algo}/${problem_set}"

# args
runs=10 # maximum runs
Gmax=1000 # maximum generations, 100 for Arm problem
record_interval=100 # generation interval for Arm problem; 1 for Arm problem
MTO=1 # 0 for STO results
popsize=100 # population size, 20 for Arm problem

# executables: [AEMTO, MATDE, SBO, MFEA2];
executable="./AEMTO" 

for tag in ${tags[@]}
do
    for tasks in ${total_tasks[@]}
    do     
        results_dir_problem="${results_dir}/${tag}_${tasks}"
        cmd="$executable \
                -problem_set ${problem_set} \
                -total_tasks $tasks_start-$tasks \
                -total_runs $runs \
                -Gmax $Gmax \
                -record_interval $record_interval \
                -popsize $popsize \
                -MTO $MTO \
                -problem_name $tag \
                -results_dir $results_dir_problem"
        echo "cmd: ${cmd}"
        $cmd
    done  
done

if [ "$?" != "0" ]
then
    echo "run failed."
    exit 1
fi
wait

# Concate the results files if the number of tasks is large (e.g. 2000)
# echo "concate results files..." 
# for tag in ${tags[@]}
# do
#     for ntask in ${total_tasks[@]}
#     do
#         echo "processing ${tag}_${ntask}"
#         results_subdir="${tag}_${ntask}"
#         results_compact="${results_dir}/${results_subdir}/compact.txt"
#         rm -f $results_compact
#         echo "start_end, ${tasks_start},${ntask}" >> $results_compact 
#         for((i=${tasks_start}; i<=${ntask}; i ++))
#         do 
#             res_file=${results_dir}/${results_subdir}/res_task_$i.json
#             cat $res_file >> $results_compact
#             rm -f $res_file
#         done
#     done
# done

# >&2 echo "done. results dir: ${results_dir}"
