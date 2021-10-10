#!/bin/bash


####
# Run an experiment on the ozStar cluster.
# First modify the cfg file, specify a experiement name,
# the results are stored at the this {Results}/{experiment}
# as well as the cfg file avoiding override.
# So you can generate multiple experiments at same time.
####

# echo "run args $*"

params_file="./cfgs/params_arm.cfg"
experiment="Arm_centers_uni/smto/DE/mto"
run_file="arm_serial.sh" # sbatch script, including the resouce specification, executable file, etc.


##################################
# shared with all problems
results_dir="Results/${experiment}"
mkdir -p $results_dir
cp $params_file $results_dir

params_file="${results_dir}/$(basename $params_file)" # using new backup cfg
echo "set backup params file ${params_file}"

cmd="sbatch $run_file $params_file $results_dir"
echo $cmd
$cmd
