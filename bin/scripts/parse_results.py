# encoding=utf-8

import json
import logging
import os
import re
import os.path as osp
import numpy as np

logging.getLogger().setLevel(logging.INFO)

DECIMALS = 8

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


def get_best_fitness(res, task_ids=[1], runs=-1, gens=-1):
    ret = {}
    for t in task_ids:
        best_fitness = [obj["fitness_values"][gens] for obj in res[t][:runs]]
        ret[t] = np.around(np.array(best_fitness), decimals=DECIMALS)
    return ret


def get_tasks_fitness(res):
    """return the raw fitness results and the min, max, fes"""
    task_ids = sorted(list(res.keys()))
    runs = len(res[task_ids[0]])
    gens = len(res[task_ids[0]][0]['fitness_values'])
    step = 1
    fes = [e for e in range(0, gens, step)]
    raw_results = []  # dims --> (tasks, runs, fitnesses)
    for task_id in task_ids:
        results_one_run = []
        for i in range(runs):
            results_one_run.append(
                res[task_id][i]['fitness_values'][:gens:step])
        raw_results.append(results_one_run)
    raw_results = np.array(raw_results)
    assert len(fes) == raw_results.shape[-1]
    return raw_results, fes
    

def get_results(results_dir, runs=-1):
    assert osp.exists(results_dir), 'results dir not exists: {}'.format(
        results_dir)
    res = read_tasks_results_from_json(results_dir)
    task_ids = sorted(list(res.keys()))
    return get_best_fitness(res, task_ids=task_ids, runs=runs)


# if __name__ == '__main__':
    # logging.info(
    #     get_MFEA_baseline_results(
    #         'Results/results_mfeas/benchmark_18_500x50_mfea.txt').keys())
    # logging.info(
    #     get_MaTDE_baseline_results(
    #         'MaTDE/Results/MaTDE_origin/matde_problem/matde_problem.txt'))
    # results_dir = sys.argv[1]
    # assert osp.exists(results_dir), 'results dir not exists: {}'.format(
    #     results_dir)
    # res = read_tasks_results_from_json(results_dir)
    # get_tasks_fitness(res)
    # results = get_results(sys.argv[1], mode='my')
    # for k, v in results.items():
    #     print('task {}, mean {:.8f}, std {:.8f}'.format(k, np.mean(v), np.std(v)))
