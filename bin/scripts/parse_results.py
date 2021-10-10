# encoding=utf-8

import json
import logging
import math
import os
import re
import sys
import os.path as osp
import numpy as np

from scipy.io import loadmat
from datetime import datetime

logging.getLogger().setLevel(logging.INFO)

DECIMALS = 6


def timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_MaTDE_baseline_results(results_file):
    assert osp.exists(results_file), 'results file {} not exists.'.format(
        results_file)
    task_results = {}
    with open(results_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            values = [
                math.fabs(float(v.strip()))
                for v in line.strip('\n').split(',')
            ]
            task_results[i + 1] = np.array(values)
    return task_results


def get_MFEA_baseline_results(results_file):
    task_results = {}
    fitness_values = []
    assert osp.exists(results_file), 'results file {} not exists.'.format(
        results_file)
    with open(results_file, 'r') as f:
        for _, line in enumerate(f.readlines()):
            values = [
                math.fabs(float(v.strip())) for v in line.strip('\n').split()
            ]
            fitness_values.append(values)

    fitness_values = np.array(fitness_values)
    ntasks = fitness_values.shape[1]

    for i in range(ntasks):
        task_results[i + 1] = fitness_values[:, i]
    return task_results


def get_multi_task_baseline(
        mat_file="data/mto_benchmark/baseline/BaselineResult.mat"):
    '''
    Get the baseline results of 9 problems
    '''
    mat = loadmat(mat_file)
    mfea_mat = mat['data_MFEA']
    soo_mat1 = mat["data_SOO_1"]
    soo_mat2 = mat["data_SOO_2"]

    baseline_mto = {}
    baseline_sto = {}
    for i in range(0, 9):
        res_two_tasks = mfea_mat[0, i]["EvBestFitness"][:, -1]
        res_t1 = res_two_tasks[::2]
        res_t2 = res_two_tasks[1::2]
        problem_id = i + 1
        baseline_mto[problem_id] = [res_t1, res_t2]

        res_t1_sto = soo_mat1[0, i]["EvBestFitness"][:, -1]
        res_t2_sto = soo_mat2[0, i]["EvBestFitness"][:, -1]

        baseline_sto[problem_id] = [res_t1_sto, res_t2_sto]
    return baseline_sto, baseline_mto


def read_tasks_results_from_json(results_dir):
    assert osp.exists(results_dir), 'path %s does not exisits' % results_dir

    parsed_results = {}
    if osp.exists(osp.join(results_dir, 'compact.txt')):
        print('read from compact file')
        all_res_file = osp.join(results_dir, 'compact.txt')
        f = open(all_res_file, 'r')
        _, start, end, = f.readline().strip().split(',')
        for task_id in range(int(start), int(end) + 1):
            json_obj = json.loads(f.readline())
            parsed_results[task_id] = json_obj
    else:
        for f in os.listdir(results_dir):
            if f.find(".json") == -1:
                continue
            task_id = int(re.search('task_.*?(\d+).*?.', f).group(1))
            json_file = osp.join(results_dir, f)
            json_obj = json.loads(open(json_file, 'r').read())
            parsed_results[task_id] = json_obj
    return parsed_results


def get_mean_solutions(res, task_ids=[1], runs=30):
    solutions = {}
    for task in task_ids:
        s = [obj["best_solutions"] for obj in res[task][:runs]]
        solutions[task] = np.mean(s, axis=0)
    return solutions


def get_best_fitness(res, task_ids=[1], runs=4, gens=-1):
    ret = {}
    for t in task_ids:
        best_fitness = [obj["fitness_values"][gens] for obj in res[t][:runs]]
        ret[t] = np.around(np.array(best_fitness), decimals=DECIMALS)
    return ret


def get_tasks_mean_fitness(res):
    all_mean_values = []
    task_id = sorted(list(res.keys()))[0]
    runs = len(res[task_id])
    step = 4
    start = 0
    gens = len(res[task_id][0]['mean_fitness'])//1
    print('runs {}, gens {}'.format(runs, gens))
    fes = [e[0] for e in res[task_id][0]['mean_fitness'][start:gens:step]]
    for i in range(runs):
        all_mean_values.append(
            [e[1] for e in res[task_id][i]['mean_fitness'][start:gens:step]])
        if len(all_mean_values[-1]) != len(fes):
            raise RuntimeError("Incomplete mean fitness recorded.")
    all_mean_values = np.array(all_mean_values)
    mean = np.mean(all_mean_values, axis=0)
    std = np.std(all_mean_values, axis=0)
    return mean, std, fes, all_mean_values


def get_tasks_fitness(res):
    """return the raw fitness results and the min, max, fes"""
    task_ids = sorted(list(res.keys()))
    runs = len(res[task_ids[0]])
    gens = len(res[task_ids[0]][0]['fitness_values'])
    step = 1
    print('runs {}, gens {}'.format(runs, gens))
    fes = [e for e in range(0, gens, step)]
    raw_results = []  # dims --> (tasks, runs, fitnesses)
    for task_id in task_ids:
        results_one_run = []
        for i in range(runs):
            results_one_run.append(
                res[task_id][i]['fitness_values'][:gens:step])
        raw_results.append(results_one_run)
    raw_results = np.array(raw_results)
    print('parse results shape {}\n fes {}'.format(
        raw_results.shape, fes))
    assert len(fes) == raw_results.shape[-1]
    return raw_results, fes


def get_my_results(results_dir, runs=13):
    assert osp.exists(results_dir), 'results dir not exists: {}'.format(
        results_dir)
    res = read_tasks_results_from_json(results_dir)
    # task_ids = list(range(1, 6))
    task_ids = sorted(list(res.keys()))
    return get_best_fitness(res, task_ids=task_ids, runs=runs)


def get_job_average_time(results, serial=False, time_index=1):
    '''
    results: parsed results
    serial: If use serial algorithm, then return the averaged summed time of
    each component task over total independent runs
    else return the averaged maximum time of all component task in each run

    time_index 1: total compute time; 2: total communication time; 3:
    communication time rate
    '''

    keys = list(results.keys())
    keys.sort()

    total_runs = len(results[keys[0]])
    total_time = 0
    total_max_time = [-1] * total_runs

    for i in range(total_runs):
        for task in keys:
            time_np = np.array(results[task])[i, time_index]
            if time_np > total_max_time[i]:
                total_max_time[i] = time_np
            total_time += time_np

    if serial:
        return total_time / total_runs
    else:
        return sum(total_max_time) / total_runs


def get_results(results, mode='my'):
    if mode is None:
        mode = results['mode']
        results_file = results['path']
    else:
        results_file = results
    if mode == 'my':
        return get_my_results(results_file)
    if mode == 'mfea':
        return get_MFEA_baseline_results(results_file)
    if mode == 'matde':
        return get_MaTDE_baseline_results(results_file)
    raise RuntimeError('unknown results type {}.'.format(mode))


if __name__ == '__main__':
    # logging.info(
    #     get_MFEA_baseline_results(
    #         'Results/results_mfeas/benchmark_18_500x50_mfea.txt').keys())
    # logging.info(
    #     get_MaTDE_baseline_results(
    #         'MaTDE/Results/MaTDE_origin/matde_problem/matde_problem.txt'))
    results_dir = sys.argv[1]
    assert osp.exists(results_dir), 'results dir not exists: {}'.format(
        results_dir)
    res = read_tasks_results_from_json(results_dir)
    get_tasks_fitness(res)
    # results = get_results(sys.argv[1], mode='my')
    # for k, v in results.items():
    #     print('task {}, mean {:.8f}, std {:.8f}'.format(k, np.mean(v), np.std(v)))
