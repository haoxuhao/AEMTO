import numpy as np
from scipy.stats import ranksums
import scipy

COMP_TAG = ('t', 'w', 'l')


def get_overall_comparisons(task_results, alpha=0.05):
    """get comparisons, using task_results[0] as reference result.
    Args:
        task_results (list): list of fitness values of multiple runs
            for each algorithm. [res1(numpy), res2,...].
        alpha (float): wilcoxon rank sum test, alpha value
    Returns:
        detailed_results ([dict]): list of results for each algorithm.
            [
                {
                'wilcoxon': 'tie/win/loss',
                'wilcoxon_details': 'tie/win/loss,w,p-value',
                'mean': ,
                'median': ,
                'std': ,
                'normalized_score': ,
                'f1_score':
                }
            ]
    """
    assert len(task_results) > 0,\
        'at least 1 task results required.'
    detailed_results = []
    ref_res = task_results[0]
    res = basic_stat_res(ref_res)
    res.update({'wilcoxon': None})
    detailed_results.append(res)
    for comp_res in task_results[1:]:
        tag, _, p_value, w = wilcoxon_test(ref_res, comp_res, alpha=alpha)
        res = basic_stat_res(comp_res)
        res.update(
            {'wilcoxon_details': '{},{},{:.3f}'.format(tag, w, p_value)})
        res.update({'wilcoxon': tag})
        detailed_results.append(res)

    for i, s in enumerate(normalized_scores(task_results)):
        detailed_results[i].update({'normalized_score': s})
    for i, f1 in enumerate(f1_scores(task_results)):
        detailed_results[i].update({'f1_score': f1})

    return detailed_results


def basic_stat_res(res):
    ret = {'mean': np.mean(res), 'median': np.median(res), 'std': np.std(res)}
    return ret


def wilcoxon_test(x1, x2, alpha=0.05):
    """ x1 is the reference one
    return [win/tie/loss] [mean1, mean2] [p-value]
    """

    if np.mean(np.abs(x1 - x2)) < 1e-8:
        tag = "t"
        p_value = 1.0
        w = 100
        mean_value = (np.mean(x1), np.mean(x2))
    else:
        mean_value = (np.mean(x1), np.mean(x2))
        is_equal = ranksums(x1, x2)
        w = is_equal[0]
        p_value = is_equal[1]

        if p_value > alpha:
            tag = "t"
        elif mean_value[0] < mean_value[1]:
            tag = "l"
        else:
            tag = "w"
    return tag, mean_value, p_value, w


def normalized_scores(results):
    """
    params results: [result1, result2, ...]
    return normalized values: [res1, res2, res3, res4]
    """
    res_np = np.array(results)
    fmax = np.max(res_np)
    fmin = np.min(res_np)

    normalized_scores = (res_np - fmin) / (fmax - fmin + 1e-12)
    normalized_scores = np.mean(normalized_scores, axis=1)

    return normalized_scores.T.tolist()


def friedman_test(mean_ranks, N=18, alpha=0.05):
    """Calculate the friedman_test
    mean_ranks: min, max, max-1, ...
    N is the total number of test cases
    Output a table of the z values p-values
    """
    k = len(mean_ranks)
    SE = (k * (k + 1.0) / (6.0 * N))**0.5
    min_rank = mean_ranks[0]
    other_ranks = mean_ranks[1:]
    sorted_indices = np.argsort(-np.array(other_ranks)).tolist()
    print('sorted indices ', sorted_indices)
    z_values = []
    p_values = []
    alphas = []
    final_results = {}
    for i in range(1, k):
        other_index = sorted_indices[i - 1] + 1
        z = (mean_ranks[other_index] - min_rank) / SE
        z_values.append(z)
        adjust_alpha = alpha / (k - i)
        alphas.append(adjust_alpha)
        p_value = scipy.stats.norm.sf(abs(z)) * 2
        p_values.append(p_value)
        final_results[other_index] = [z, p_value, adjust_alpha]

    sorted_keys = sorted(list(final_results.keys()))
    for k in sorted_keys:
        z, p_value, adjust_alpha = final_results[k]
        print(
            '{}, z_value: {:.5f}; p_value: {:.5f}; adjust alpha {:5f}'.format(
                k, z, p_value, adjust_alpha))


def f1_scores(results):
    """ calculate the f1 scores of each algorithm
    params results: [result1, result2, ...]
    return score values: [res1, res2, res3, res4]
    """
    Valid_Score = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] # according to F1 race
    res_np = np.array(results)
    scores = [0] * len(results)
    res_np_median = np.median(res_np, axis=1)
    ranks = np.argsort(res_np_median.T)
    for j, r in enumerate(ranks):
        scores[r] = Valid_Score[j]
    return scores


def mean_rank_scores(results):
    """ calculate the mean rank values scores of each algorithm
    params results: [result1, result2, ...]
    return score values: [res1, res2, res3, res4]
    """
    res_np = np.array(results)
    scores = [0] * len(results)
    res_np_mean = np.mean(res_np, axis=1) # use mean rank
    ranks = np.argsort(res_np_mean.T)
    for j, r in enumerate(ranks):
        scores[r] = j + 1 # mean rank value
    return scores


# if __name__ == '__main__':
#     pass
    # test suit 3
    # mean_ranks = [a/6.0 for a in [10.72, 23.60, 16.14, 26.54, 21.00, 28.00]]
    # test suite 2
    # mean_ranks = [a/4.0 for a in [5.20, 9.10, 13.50, 12.20]]
    # test suit 1
    # mean_ranks = [a/9.0 for a in [13.50, 20.50, 20.00]]

    # ranks_testsuite1 = np.array([23.50, 30.00, 28.50, 37.50, 38.50,
    #                              31.00]) * 2  # 9
    # ranks_manytask10 = np.array([1.5, 3.75, 3.175, 4.125, 3.95, 4.5]) * 40  # 4
    # ranks_matde_problem = np.array([1.20, 4.70, 2.90, 4.80, 3.70, 3.70]) * 10  # 1
    # ranks_cec50 = np.array([10.72, 23.60, 16.14, 26.54, 21.00, 28.00]) * 50  # 6
    # total_mean_ranks = (ranks_cec50 + ranks_manytask10 +
    #                     ranks_matde_problem + ranks_testsuite1) / (18 + 50 + 300)
    # print(['%.2f' % x for x in total_mean_ranks])

    # print(mean_ranks)
    # mean_ranks = [3.0, 3.7, 4.5, 3.9, 4.1, 4.6, 4.2]
    # mean_ranks = [1.44, 1.56, 3.0, 4.0, 6.36, 5.0, 6.64]
    # mean_ranks = [1.10, 1.89]
    # friedman_test(mean_ranks, N=300, alpha=0.05)
