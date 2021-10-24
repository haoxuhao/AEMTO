# !/bin/env python
import copy
import numpy as np
import pandas as pd

from eval import get_overall_comparisons, COMP_TAG
from parse_results import get_results, DECIMALS


def overall_compare(results, runs=15, algos=None):
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
    if algos is None:
        algo_prefix = 'algo_{}_'
    else:
        assert algorithms == len(algos)

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
            if algos is None:
                prefix = algo_prefix.format(i)
            else:
                prefix = algos[i] + '_'
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
        if algos is None:
            prefix = algo_prefix.format(i)
        else:
            prefix = algos[i] + '_'
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
        save_file='./overall_comparisons.csv',
        algos=None):
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
            results[p], runs=runs, algos=algos)
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
    print('save detailed comparison to file {}'.format(save_file))

    return (total_summeray, normalized_scores_total, f1_scores_total)


def build_comparisons():
    algos = ['AEMTO', 'SBO', 'MFEA', 'MATDE', 'STO'] # STO is AEMTO --MTO 0
    problem_set = 'matde_problem'
    problem_name = "zero_10"
    compared_res = {
        problem_name : [get_results('Results/{}/{}/{}'.format(
            algo, problem_set, problem_name)) for algo in algos]
    }
    return compared_res, algos


if __name__ == "__main__":
    results_to_compare, algo_labels = build_comparisons()
    total_summeray, normalized_scores_total, \
        f1_scores_total = summeray_compare(results_to_compare, runs=10, algos=algo_labels)
    print('\n========================== summeray =========================\n')
    print(" .  wilcoxon total summeray: ")
    for i, item in enumerate(total_summeray):
        item_str = '{}/{}/{}'.format(item['t'], item['w'], item['l'])
        print(' .  algo {}: total comparisons to algo {} ties/wins/losses: {}'.
              format(algo_labels[i+1], algo_labels[0], item_str))
    print(" .  normalized scores: {}".format(', '.join(
        ['%.4f' % a for a in normalized_scores_total])))
    print(" .  f1 scores: {}".format(', '.join(
        ['%.2f' % a for a in f1_scores_total])))
    print('\n=============================================================')
