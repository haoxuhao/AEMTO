import os
import os.path as osp

from parse_results import (read_tasks_results_from_json, get_tasks_fitness)
from draw_convergence import draw_normalized_scores_convergence


colors = ['r', 'g', 'b', 'orange', 'cyan', 'purple', 'k', 'm', 'olive']
markers = ["*", "x", "o", "s", ">", "*", "x", "o", "s", ">"]


def convgence_compare():
    problem_name = 'Arm'
    results_dirs = [
        'Results/{}/{}',
        'Results/{}/{}',
        'Results/MFEA/{}/{}',
        'Results/SBO/{}/{}',
        'Results/MaTDE/{}/{}',
    ]
    labels = ['STO', 'AEMTO', 'MFEA2', 'SBO', 'MaTDE']
    save_dir = '{}'.format(problem_name)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    ntasks = 2000
    problems = [
        'zero_{}'.format(ntasks),
    ]
    for problem in problems:
        results = []
        for res_dir in results_dirs:
            res_json = read_tasks_results_from_json(
                res_dir.format(problem_name, problem))
            results.append(get_tasks_fitness(res_json))
        save_file = osp.join(save_dir,
                             'arm_{}_normalized_scores_comparison.pdf'.format(problem))
        draw_normalized_scores_convergence(results,
                                           labels=labels,
                                           save_file=save_file)

if __name__ == '__main__':
    convgence_compare()
