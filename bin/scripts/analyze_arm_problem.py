import os
import sys
import math
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from parse_results import (get_tasks_mean_fitness,
                           read_tasks_results_from_json, get_tasks_fitness)
from eval import wilcoxon_test
from draw_convergence import draw_normalized_scores_convergence


colors = ['r', 'g', 'b', 'orange', 'cyan', 'purple', 'k', 'm', 'olive']
markers = ["*", "x", "o", "s", ">", "*", "x", "o", "s", ">"]


def draw_length_angle_range():
    problem_name = 'centroids_2000_2'  # centroids_5000_2
    # data_file = 'data/Arm/uniform_centers/'\
    #             'cluster50_center20_D2/{}.dat'.format(problem_name)
    data_file = 'data/Arm/{}.dat'.format(problem_name)
    # data_file = '../related_code/mapelit/{}.dat'.format(problem_name)
    save_dir = 'tmp/arm-centers/'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    save_file = osp.join(save_dir, '{}.pdf'.format(problem_name))
    font2 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 20,
    }
    data = np.loadtxt(data_file)

    plt.figure(figsize=(8, 6.5))
    plt.scatter((data[:, 0] - 0.5) * math.pi, data[:, 1], marker='.')
    plt.xlabel(r'$\alpha_{max}$', font2)
    plt.ylabel(r'$L$', font2)
    plt.xticks(fontproperties='Times New Roman', fontsize=14)
    plt.yticks(fontproperties='Times New Roman', fontsize=14)
    plt.savefig(save_file, dpi=300)
    print('save to file {}'.format(save_file))


def draw_selection_intensity(
    res_json,
    save_file='tmp/selection probs/Arm/centroid-{}.pdf'
):
    problem_name = 'centroids_2000_2'
    data_file = 'data/Arm/{}.dat'.format(problem_name)
    centroids = np.loadtxt(data_file)
    task_id_to_show = 1999
    save_file = save_file.format(task_id_to_show)
    runs = len(res_json[task_id_to_show])
    task_selection_probs = []
    for run_id in range(runs):
        selection_probs = res_json[task_id_to_show][run_id]['selection_probs']
        task_selection_probs.append(selection_probs)
    mean_selections = np.mean(np.array(task_selection_probs),
                              axis=(0, 1)).tolist()
    topk_show = -1
    scale = 100
    draw_map = np.zeros((scale, scale))
    target = int(centroids[task_id_to_show, 0]*scale), int(centroids[task_id_to_show, 1]*scale) 
    print('target {}'.format(target))
    sorted_indices = sorted(range(len(mean_selections)),
                            key=mean_selections.__getitem__,
                            reverse=True)
    for i in sorted_indices[:topk_show]:
        val = mean_selections[i]
        x, y = int(scale * centroids[i, 0]), int(scale * centroids[i, 1])
        draw_map[x, y] = val
    selections_sorted = sorted(mean_selections, reverse=True)
    print('top 20 selections', selections_sorted[:20])
    print('top 20 selections sum {}'.format(sum(selections_sorted[:20])))
    print('top 50 selections sum {}'.format(sum(selections_sorted[:50])))
    print('top 100 selections sum {}'.format(sum(selections_sorted[:100])))
    print('top 200 selections sum {}'.format(sum(selections_sorted[:200])))
    plt.figure(num=0)
    plt.imshow(draw_map)
    plt.colorbar()
    file_dir = osp.dirname(save_file)
    if not osp.exists(file_dir):
        os.makedirs(file_dir)
    plt.savefig(save_file)
    print('save file to {}'.format(save_file))


def read_mapelit_results_archive(run_start,
                                 run_end,
                                 res_run_template='../related_code/mapelit/'
                                 'archive/benchmark/zero_2000/run{}/'):
    fes = []
    res = []  # dim (runs, tasks, gens)
    for run_id in range(run_start, run_end + 1):
        res_dir = res_run_template.format(run_id)
        filenames = os.listdir(res_dir)
        fes = sorted([int(s.split('.')[0].split('_')[-1]) for s in filenames])
        data = []
        for fe in fes:
            filename = 'archive_{}.dat'.format(fe)
            tmp = np.loadtxt(osp.join(res_dir, filename))
            data.append(-tmp[:, 1].T)
        data = np.array(data)
        res.append(data.T.tolist())
    res_np = np.array(res)
    res_np = res_np.transpose(1, 0, 2)  # tasks, runs, gens
    fes = np.array(fes) / 250000
    print('mape min {}'.format(np.min(res_np)))
    print('read mape result from {}; shape {}'.format(res_run_template,
                                                      res_np.shape))
    return res_np, fes


def read_mapelit_results(run_start,
                         run_end,
                         mapelit_res_file_template='../related_code/mapelit/'
                         'conv_save_tasks_2000_D_50/cover_max_mean_run{}.dat'):
    fitness_all = []
    fes = []
    step = 4
    for run_id in range(run_start, run_end + 1):
        res_file = mapelit_res_file_template.format(run_id)
        data = np.loadtxt(res_file)
        fes = data[:, 0]
        gens = data[:, 0].shape[0] // 1
        fes = fes[:gens:step]
        fitness_all.append(data[:, 3][:gens:step])

    fitness_all = np.array(fitness_all)
    mean = -np.mean(fitness_all, axis=0)
    std = -np.std(fitness_all, axis=0)
    return mean, std, fes


def get_wilcoxon_test_results(results, labels):
    """results: [get_tasks_mean_fitness(), ...]
       sequential comparing 2 results
    """
    assert len(labels) == len(results)
    n_algos = len(results)
    comparing_res = {}
    for i in range(n_algos - 1):
        tag = '{} <- {}'.format(labels[i], labels[i + 1])
        res_a = results[i][-1][:, -1]
        res_b = results[i + 1][-1][:, -1]
        comparing_res[tag] = wilcoxon_test(res_a, res_b)
    for k, v in comparing_res.items():
        print(k, v)
    return comparing_res


def draw_convergence(
    results,
    labels,
    save_file='tmp/tmp.pdf',
):
    """ results: [[mean, std, fes], [mean, std, fes],...]
    """
    assert len(labels) == len(results)
    # styles = ['k-.', 'k-.', 'k-.', 'k-.', 'k-.', 'k-', 'k-', 'k-', 'k-', 'k-']

    markers = ['-']
    font2 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 22,
    }
    # colors = ['r', 'g', 'b', 'orange', 'cyan', 'purple']
    plt.figure(figsize=(8, 6.5))
    scale = 1000
    for i, res in enumerate(results):
        x = np.array(res[2]) / scale
        print('res {}, best mean values {}'.format(i, res[0][-1]))
        plt.plot(x,
                 res[0],
                 color=colors[i % len(colors)],
                 label=labels[i],
                 marker=markers[i % len(markers)])
        plt.fill_between(x,
                         res[0] - res[1],
                         res[0] + res[1],
                         color=colors[i % len(colors)],
                         alpha=0.2)
    plt.xlabel('FEs (x%d)' % (scale), font2)
    plt.ylabel('Mean fitness', font2)
    plt.xticks(fontproperties='Times New Roman', fontsize=16)
    plt.yticks(fontproperties='Times New Roman', fontsize=16)
    plt.legend(prop={'family': 'Times New Roman', 'size': 16})
    plt.savefig(save_file, dpi=300)
    print('save file to {}'.format(save_file))
    plt.close()


def convgence_compare():
    problem_name = 'ArmDim50'
    results_dirs = [
        'Results/{}/smto/DE/sto/{}',
        'Results/{}/smto/DE/mto_fixed_tsf_uni_sel/{}',
        'Results/{}/smto/DE/mto_ada_tsf_uni_sel/{}',
        # 'Results/{}/smto/DE/mto_ada_tsf_ada_sel/{}',
        # 'Results/{}/smto/DE/mto_ada_tsf_ada_sel_top50_sample/{}',
        # 'Results/{}/smto/DE/mto_ada_tsf_ada_sel_top100_sample/{}',
        # 'Results/{}/smto/DE/sto/{}',
        # 'MFEA/Results/{}/smto/DE/mto/{}',
        # 'SBO/Results/{}/smto/DE/mto_v2/{}',
        # 'MaTDE/Results/{}/smto/DE/mto2/{}',
        'Results/{}/smto/DE/mto_ada_tsf_ada_sel/{}',
        # 'MaTDE/Results/{}/smto/DE/sto3/{}',
    ]
    labels = [
        # 'AEMTO',
        'STO',
        'AEMTO (w/o aTsf and aSel)',
        'AEMTO (w/o aSel)',
        # 'AEMTO',
        # 'AEMTO (top-50 sample)',
        # 'AEMTO(top-100 sample)',
        # 'STO',
        # 'MFEA2',
        # 'SBO',
        # 'MaTDE',
        'AEMTO',
    ]

    save_dir = 'tmp/convergence/{}'.format(problem_name)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    res_mapelit = read_mapelit_results_archive(1, 10)
    # res_mapelit = read_mapelit_results(1, 3)

    ntasks = 2000
    problems = [
        'zero_{}'.format(ntasks),
        # 'small_{}'.format(ntasks),
        # 'median_{}'.format(ntasks)
    ]

    for problem in problems:
        results = []
        for res_dir in results_dirs:
            res_json = read_tasks_results_from_json(
                res_dir.format(problem_name, problem))
            # results.append(get_tasks_mean_fitness(res_json))
            results.append(get_tasks_fitness(res_json))

        # save_file = osp.join(save_dir, '{}_mean_fitness.pdf'.format(problem))
        # draw_convergence(results, labels=labels, save_file=save_file)
        # get_wilcoxon_test_results(results, labels=labels)

        # results.append(res_mapelit)
        # labels.append('MAP-Elite')
        save_file = osp.join(save_dir,
                             '{}_normalized_scores_modular.pdf'.format(problem))
        draw_normalized_scores_convergence(results,
                                           labels=labels,
                                           save_file=save_file)


def time_compare():
    font2 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 28,
    }
    fig = plt.figure(figsize=(8, 6.5))
    # fig, ax = plt.subplots()
    # fig = plt.figure(figsize=(8, 6.5))
    # left, bottom, width, height (range 0 to 1)
    left, bottom = 0.20, 0.15
    ax = fig.add_axes([left, bottom, (1 - left) * 0.9, (1 - bottom) * 0.9])
    bar_x = [1, 2, 3, 4, 5]
    bar_height = [13.3, 12.9, 14.9, 715.3, 192.4]
    bar_tick_label = ['AEMTO', 'STO', 'SBO', 'MaTDE', 'MFEA2']
    bar_label = [13.3, 12.9, 14.9, 715.3, 192.4]
    # ax.grid(zorder=1)

    plt.grid(axis="y")
    bar_plot = plt.bar(bar_x,
                       bar_height,
                       tick_label=bar_tick_label,
                       color='darkorange')
    plt.ylim(0, 800)

    def autolabel(rects):
        for idx, rect in enumerate(bar_plot):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.,
                    1.05 * height,
                    bar_label[idx],
                    ha='center',
                    va='bottom',
                    rotation=0,
                    fontdict={
                        'family': 'Times New Roman',
                        'weight': 'normal',
                        'size': 20,
                    })

    autolabel(bar_plot)
    plt.xticks(fontproperties='Times New Roman', fontsize=22)
    plt.yticks(fontproperties='Times New Roman', fontsize=22)
    plt.ylabel('computation time (min)', font2)
    save_file = 'tmp/time.pdf'
    # plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    print('save file to {}'.format(save_file))


if __name__ == '__main__':
    # draw_length_angle_range()
    convgence_compare()
    time_compare()
    # results_dir = 'Results/ArmDim50/smto/DE/mto_vis/zero_2000'
    # draw_selection_intensity(read_tasks_results_from_json(results_dir))
