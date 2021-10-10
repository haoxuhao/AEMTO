# coding=utf8
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import os

from matplotlib.pyplot import MultipleLocator
from parse_results import (read_tasks_results_from_json,
                           get_tasks_fitness)

matplotlib.rcParams['axes.unicode_minus'] = False

colors = ['r', 'g', 'b', 'orange', 'cyan', 'purple', 'k', 'm', 'olive']
markers = ["*", "x", "o", "s", ">", "*", "x", "o", "s", ">"]


def calculate_grids(ntasks):
    if ntasks <= 0:
        return [0, 0]

    a = ntasks**0.5
    floor_a = int(math.floor(a))
    ceil_a = int(math.ceil(a))

    if floor_a == ceil_a:
        grid = [ceil_a, ceil_a]
    elif floor_a * ceil_a >= ntasks:
        grid = [floor_a, ceil_a]
    else:
        grid = [ceil_a, ceil_a]
    return grid


def draw_convergence_one_problem_multi_algorithms(
        results,
        labels=["1", "2", "3", "4", "5"],
        colors=["red", "green", "blue", "black"],
        markers=["D", "x", "o", ">", "s", ".", "D"],
        task_ids=None,
        figure_index=0,
        with_marks=True,
        save_file='./tmp/convergece.png'):
    """plot the convergence rate of one problem over different algorithms
       Only suitable for two or three tasks each problem
    """
    assert len(task_ids) <= 3, 'two or three tasks each problem only'
    styles = ['k-.', 'r-', 'b--']
    fontdict = {'family': 'Times New Roman', 'size': 28}
    # plt.figure(figure_index, figsize=(8, 6.5))
    fig = plt.figure(figsize=(8, 6.5))
    # left, bottom, width, height (range 0 to 1)
    left, bottom = 0.2, 0.15
    ax = fig.add_axes([left, bottom, (1 - left) * 0.9, (1 - bottom) * 0.9])
    for i, task_id in enumerate(task_ids):
        line_style = styles[i]
        for j, res in enumerate(results):
            marker = markers[j]
            fitness = [obj["fitness_values"] for obj in res[task_id]]
            generations = np.array(res[task_id][0]["generations"])[1:]
            averaged_fitness = np.log10(np.mean(fitness, axis=0)[1:])
            plt.plot(generations,
                     averaged_fitness,
                     line_style,
                     label=labels[j] + r' ($\tau_%d$)' % task_id,
                     marker=marker)
    plt.legend(loc=0, prop={'family': 'Times New Roman', 'size': 20})
    plt.xlabel('$g$', fontdict=fontdict)
    plt.ylabel('$log_{10}(FEV)$', fontdict=fontdict)
    plt.xticks(fontproperties='Times New Roman', fontsize=24)
    plt.yticks(fontproperties='Times New Roman', fontsize=24)
    # y_major_locator = MultipleLocator(1)
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(y_major_locator)
    plt.savefig(save_file)
    plt.close()
    print('save image file to {}'.format(save_file))


def draw_averaged_convergence(results,
                              labels=["1", "2", "3", "4", "5"],
                              colors=["red", "green", "blue", "black"],
                              markers=["D", "x", "o", ">", "s", ".", "D"],
                              task_ids=None,
                              figure_index=0,
                              with_marks=True,
                              save_file='./tmp/convergece.png'):
    """ Plot convergence rate
    results (list): list of results of multiple different settings
    labels (list): setting names
    task_ids (list): None means show all tasks
    """
    styles = ['k-.', 'k-.', 'k-.', 'k-.', 'k-.', 'k-.', 'k-.', 'k-.', 'k-.']
    styles = ['r-.', 'k-', 'b--', 'g:']
    fontdict = {'family': 'Times New Roman', 'size': 28}
    # calculate grids
    if task_ids:
        ntasks = len(task_ids)
        task_ids = task_ids
    else:
        task_ids = [int(key) for key in results[0].keys()]
        task_ids.sort()
        ntasks = len(task_ids)

    grid = calculate_grids(ntasks)

    # draw averaged convergence
    # plt.figure(figure_index, figsize=(8, 6.5))
    fig = plt.figure(figsize=(8, 6.5))
    # left, bottom, width, height (range 0 to 1)
    left, bottom = 0.2, 0.15
    ax = fig.add_axes([left, bottom, (1 - left) * 0.9, (1 - bottom) * 0.9])
    # plt.figure(figsize=(5, 4))
    k = 0
    for i in range(grid[0]):
        for j in range(grid[1]):
            if (k == len(task_ids)):
                break
            curr_task = task_ids[k]

            # plt.subplot(grid[0], grid[1], k + 1)
            # plt.subplots_adjust(wspace=0.8, hspace=0.8)
            # plt.title("$T_{%s}$" % str(curr_task), fontsize=14,
            #           fontdict=fontdict)

            for i, res in enumerate(results):
                fitness = [obj["fitness_values"] for obj in res[curr_task]]
                generations = np.array(res[curr_task][0]["generations"])[1:]
                averaged_fitness = np.log10(np.mean(fitness, axis=0)[1:])
                if with_marks:
                    plt.plot(generations,
                             averaged_fitness,
                             styles[i % len(styles)],
                             label=labels[i],
                             marker=markers[i % len(markers)])
                else:
                    plt.plot(generations,
                             averaged_fitness,
                             styles[i % len(styles)],
                             label=labels[i])

            plt.legend(loc=0, prop={'family': 'Times New Roman', 'size': 20})
            plt.xlabel('$g$', fontdict=fontdict)
            plt.ylabel('$log_{10}(FEV)$', fontdict=fontdict)
            k += 1
    plt.xticks(fontproperties='Times New Roman', fontsize=24)
    plt.yticks(fontproperties='Times New Roman', fontsize=24)
    plt.savefig(save_file, dpi=300)
    print("save image file to {}".format(save_file))
    plt.close()


def draw_normalized_scores_convergence(
    results,
    labels,
    save_file='tmp/tmp.pdf',
):
    """results: [get_tasks_fitness(), ...]"""
    assert len(labels) == len(results)
    '''
        get the normalized scores along generations
        f_n = (f_i - f_min) / (f_max - f_min)
        f_min and f_max are the min and max values
        among different algorithms along generations
        over different runs of task i
    '''
    ntasks = results[0][0].shape[0]
    f_min = [1e10] * ntasks
    f_max = [-1] * ntasks
    for i in range(ntasks):
        # among different algorithms
        for res, _ in results:
            # along generations over different runs
            f_min[i] = min(f_min[i], np.min(res[i, :, :]))
            f_max[i] = max(f_max[i], np.max(res[i, :, :]))
    f_min = np.array(f_min).reshape(-1, 1, 1)
    f_max = np.array(f_max).reshape(-1, 1, 1)
    print(np.min(f_min))
    print(np.min(f_max))
    font2 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 28,
    }
    print_interval = 1
    show_interval = 10
    # plt.figure(figsize=(8, 6.5))
    fig = plt.figure(figsize=(8, 6.5))
    # left, bottom, width, height (range 0 to 1)
    left, bottom = 0.15, 0.15
    ax = fig.add_axes([left, bottom, (1 - left) * 0.94, (1 - bottom) * 0.9])
    for i, res in enumerate(results):
        fitness, fes = res
        print('{} best mean fitness {}'.format(labels[i],
                                               np.mean(fitness[:, :, -1])))
        fitness = (fitness - f_min) / (f_max - f_min + 1e-12)
        fitness_along_gens = np.mean(fitness, axis=(0, 1))
        print(i, fitness_along_gens)
        plt.plot(np.array(fes)[::show_interval] * print_interval,
                 fitness_along_gens[::show_interval],
                 color=colors[i % len(colors)],
                 label=labels[i],
                 marker=markers[i % len(markers)])
    plt.xlabel('$g$', font2)
    plt.ylabel('mean normalized score', font2)
    plt.xticks(fontproperties='Times New Roman', fontsize=22)
    plt.yticks(fontproperties='Times New Roman', fontsize=22)
    plt.legend(prop={'family': 'Times New Roman', 'size': 20})
    plt.savefig(save_file, dpi=300)
    print('save file to {}'.format(save_file))
    plt.close()


if __name__ == "__main__":
    # matde
    results_dirs = [
        'Results/matde_problem/smto/DE/deepinsight/mto/{}',
        'Results/matde_problem/smto/DE/deepinsight/mto_ada_tsf_uni_sel/{}',
        'Results/matde_problem/smto/DE/deepinsight/mto_fixed_tsf_uni_sel/{}',
        'Results/matde_problem/smto/DE/deepinsight/sto/{}'
    ]
    labels = ['AEMTO', 'AEMTO (w/o aSel)',
              'AEMTO (w/o aTsf and aSel)', 'STO']
    task_ids = range(1, 11)
    problems = ['zero_10']
    problem_save_root = 'matde_problem'
    # mto benchmark
    results_dirs = [
        'Results/benchmark/smto/DE/deepinsight/mto/{}',
        'Results/benchmark/smto/DE/deepinsight/mto_fixed_tsf/{}',
        'Results/benchmark/smto/DE/deepinsight/sto/{}',
    ]
    labels = ['AEMTO', 'AEMTO (w/o aTsf)', 'STO']
    task_ids = range(1, 3)
    problems = ['problem{}_2'.format(id) for id in range(1, 10)]
    problem_save_root = 'mtobenchmark'

    # manytask10
    # results_dirs = [
    #     'Results/ManyTask10/smto/DE/deepinsight/mto/{}',
    #     'Results/ManyTask10/smto/DE/deepinsight/mto_ada_tsf_uni_sel/{}',
    #     'Results/ManyTask10/smto/DE/deepinsight/mto_fixed_tsf_uni_sel/{}',
    #     'Results/ManyTask10/smto/DE/deepinsight/sto/{}'
    # ]
    # labels = ['AEMTO', 'AEMTO (w/o aSel)',
    #           'AEMTO (w/o aTsf and aSel)', 'STO']
    # task_ids = range(1, 11)
    # tags = ['zero', 'small', 'median', 'large']
    # problems = ['{}_10'.format(tag) for tag in tags]
    # problem_save_root = 'manytask10'

    save_root = 'tmp/convergence/{}/{}'
    assert len(labels) == len(results_dirs)
    for show_tag in problems:
        save_dir = save_root.format(problem_save_root, show_tag)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        results = [
            read_tasks_results_from_json(
                res.format(show_tag))
            for res in results_dirs
        ]
        ''' draw normalized scores convergence'''
        # fitness_results = [
        #     get_tasks_fitness(res)
        #     for res in results
        # ]
        # save_file = osp.join(save_dir, 'norm_conv_{}.pdf'.format(show_tag))
        # draw_normalized_scores_convergence(fitness_results, labels, save_file)

        for task in task_ids:
            save_file = osp.join(save_dir, 'conv_{}.pdf'.format(task))
            draw_averaged_convergence(
                results, labels=labels, task_ids=[task],
                save_file=save_file)
        if problem_save_root == 'mtobenchmark':
            save_file = osp.join(save_dir, 'conv_{}.pdf'.format(show_tag))
            draw_convergence_one_problem_multi_algorithms(results,
                                                          labels=labels,
                                                          save_file=save_file,
                                                          task_ids=task_ids)
