# !/bin/env python
# -*- coding: UTF-8 -*-

import copy
import numpy as np
import pandas as pd
import sys

from eval import get_overall_comparisons, COMP_TAG
from parse_results import get_results, DECIMALS
from tqdm import tqdm


def overall_compare(results, runs=15):
    """ overall comparing results of several algorithms.
    Args:
        results (list): list of results of each algorithm.
    Returns:
        overall_ranksum_test_res (dict): {"tie": 0, "loss": 0, "win": 0}
        averaged_normalized_scores (list)
        averaged_f1_scores (list)
        details_df (pd.DataFrame)
    """
    assert len(results) > 0, 'at least 1 result required.'
    algorithms = len(results)
    overall_stat_summary = {COMP_TAG[0]: 0, COMP_TAG[1]: 0, COMP_TAG[2]: 0}
    overall_ranksum_test_res = [
        copy.deepcopy(overall_stat_summary) for i in range(algorithms)
    ]

    averaged_normalized_scores = [0 for i in range(algorithms)]
    averaged_f1_scores = [0 for i in range(algorithms)]

    tasks = sorted(list(results[0].keys()))
    details_data = []
    column_names = []
    algo_prefix = 'algo_{}_'
    for task in tasks:
        task_results = [
            np.around(res[task][:runs], decimals=DECIMALS) for res in results
        ]
        overall_results = get_overall_comparisons(task_results)

        # generate detailed results
        row_dict = {}
        row_dict['task'] = task
        column_names = ['task']
        eval_items = [
            'mean', 'wilcoxon', 'normalized_score', 'f1_score', 'median', 'std'
        ]
        for i, res in enumerate(overall_results):
            prefix = algo_prefix.format(i)
            for item_name in eval_items:
                if item_name not in [
                        'wilcoxon', 'f1_score', 'normalized_score'
                ]:
                    row_dict[prefix + item_name] = '%.2E' % res[item_name]
                else:
                    row_dict[prefix + item_name] = res[item_name]
                column_names.append(prefix + item_name)
            if i != 0:
                overall_ranksum_test_res[i][res['wilcoxon']] += 1
            averaged_normalized_scores[i] += res['normalized_score']
            averaged_f1_scores[i] += res['f1_score']
        details_data.append(row_dict)

    def avg_func(x):
        return x / len(tasks)

    averaged_f1_scores = list(map(avg_func, averaged_f1_scores))
    averaged_normalized_scores = list(map(avg_func,
                                          averaged_normalized_scores))
    row_dict = {}
    for i in range(algorithms):
        prefix = algo_prefix.format(i)
        if i != 0:
            overall_res_str = ''
            for k, v in overall_ranksum_test_res[i].items():
                overall_res_str += '%s:%d/' % (k, v)
            row_dict[prefix + 'wilcoxon'] = overall_res_str

        row_dict[prefix + 'normalized_score'] = averaged_normalized_scores[i]
        row_dict[prefix + 'f1_score'] = averaged_f1_scores[i]
    details_data.append(row_dict)

    detailes_df = pd.DataFrame(data=details_data, columns=column_names)

    return (overall_ranksum_test_res, averaged_normalized_scores,
            averaged_f1_scores, detailes_df)


def summeray_compare(
        results,
        runs=15,
        save_file='./tmp/detailed_results/overall_comparisons.csv'):
    """ compare several paired results
    Args:
        results (dict): {'problem_name': [res1, res2]}
        runs (int): runs to compare
    Returns:
       total_summeray
       normalized_scores_total
       detailed_summeray
    """
    problems = sorted(list(results.keys()))
    algo_num = len(results[problems[0]])

    total_summeray = {}
    normalized_scores_total = {}
    f1_scores_total = {}

    total_summeray = [
        copy.deepcopy({
            COMP_TAG[0]: 0,
            COMP_TAG[1]: 0,
            COMP_TAG[2]: 0
        }) for i in range(algo_num - 1)
    ]
    normalized_scores_total = [0] * algo_num
    f1_scores_total = [0] * algo_num

    total_detailed_res_df = pd.DataFrame()

    for p in problems:
        wilcx_res, norm_scores, f1_scores_, detailed_res_df = overall_compare(
            results[p], runs=runs)
        total_detailed_res_df = total_detailed_res_df.append([{'task': p}])
        total_detailed_res_df = total_detailed_res_df.append(detailed_res_df)

        print('', p, '\n normalized scores:', norm_scores, '\n f1_scores:',
              f1_scores_)
        for i in range(algo_num):
            res = wilcx_res[i]
            if i != 0:
                for tag in COMP_TAG:
                    total_summeray[i - 1][tag] += res[tag]
                wilcx_res_str = '{}/{}/{}'.format(res['t'], res['w'], res['l'])
                print(' algo {} wilcoxon res ties/wins/losses {}'.format(
                    i, wilcx_res_str))

            normalized_scores_total[i] += norm_scores[i]
            f1_scores_total[i] += f1_scores_[i]

    total_detailed_res_df.to_csv(save_file)

    return (total_summeray, normalized_scores_total, f1_scores_total)


def build_comparisons():
    res_dirs = [
        'Results/{}/smto/DE/deepinsight/mto/{}',
        'Results/{}/smto/DE/deepinsight/mto_fixed_tsf/{}',
        'MaTDE/Results/{}/smto/DE/mto/{}',
        'SBO/Results/{}/smto/DE/mto/{}',
        'MFEA/Results/{}/smto/DE/mto/{}',
        'Results/{}/smto/DE/deepinsight/sto/{}',
    ]

    # tags = ['zero_10']
    # problem_name = 'matde_problem'

    tags = ['problem{}_2'.format(i) for i in range(1, 10)]
    problem_name = 'benchmark'

    compared_res = {}
    for tag in tqdm(tags):
        print('load algorithm results from problem: {}'.format(tag))
        compared_res[tag] = [
            get_results(res.format(problem_name, tag)) for res in res_dirs
        ]
    return compared_res

def ci():
    matde_res_dirs = [
        'Results/matde_problem_ref/zero_10',
        'Results/matde_problem/DE/mto/zero_10',
    ]
    arm_res_dirs = [
        'Results/ArmD50_Ref/zero_10',
        'Results/Arm/DE/mto/zero_10',
    ]
    compared_res = {
        'arm_zero_10': [get_results(res) for res in arm_res_dirs],
        'matde_zero_10': [get_results(res) for res in matde_res_dirs]
    }
    return compared_res


if __name__ == "__main__":
    results_to_compare = ci()

    total_summeray, normalized_scores_total, \
        f1_scores_total = summeray_compare(results_to_compare, runs=20)

    print('\n========================== summeray =========================\n')
    print(" .  wilcoxon total summeray: ")
    for i, item in enumerate(total_summeray):
        item_str = '{}/{}/{}'.format(item['t'], item['w'], item['l'])
        print(' .  algo {}: total comparisons to algo 0  ties/wins/losses: {}'.
              format(i + 1, item_str))
    print(" .  normalized scores: {}".format(', '.join(
        ['%.4f' % a for a in normalized_scores_total])))
    print(" .  f1 scores: {}".format(', '.join(
        ['%.2f' % a for a in f1_scores_total])))
    print('\n=============================================================')
