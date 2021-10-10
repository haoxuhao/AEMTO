import json
import math
import os
import sys

import os.path as osp
import matplotlib.pyplot as plt
import numpy as np

from parse_results import read_tasks_results_from_json


def draw_from_results(results_file, save_file='tmp/tmp.pdf', task_ids=[1]):
    """results read from json"""
    results = read_tasks_results_from_json(results_file)
    import_prob_task = {}
    runs = len(results[task_ids[0]])
    intervals = 50
    font2 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 28,
    }
    markers = ['>', 'o']
    # colors = ['blue', 'read']
    # plt.figure(figsize=(8, 6.5))
    fig = plt.figure(figsize=(8, 6.5))
    # left, bottom, width, height (range 0 to 1)
    left, bottom = 0.2, 0.15
    ax = fig.add_axes([left, bottom, (1 - left) * 0.9, (1 - bottom) * 0.9])
    x_values = list(range(0, 1000))[0::intervals]
    for i, task_id in enumerate(task_ids):
        import_probs = []
        for run in range(runs):
            import_prob = list(results[task_id][run]['import_prob'])
            import_probs.append(import_prob)
        import_probs_np = np.array(import_probs)
        mean_values = np.mean(import_probs_np, axis=0)[0::intervals]
        std_values = np.std(import_probs_np, axis=0)[0::intervals]
        import_prob_task[task_id] = mean_values

        x_values = list(range(0, 1000))[0::intervals]
        plt.errorbar(x_values, list(import_prob_task[task_id]),
                     yerr=std_values, fmt='o-', ecolor='k',
                     color='k', elinewidth=2, capsize=4,
                     label='adaptive $p_{%d}^{tsf}$' % task_id,  # % task_id
                     marker=markers[i])
        # plt.plot(x_values, list(import_prob_task[task_id]),
        #          label='$T_{%d}$' % task_id)

        # plt.fill_between(x_values, import_prob_task[task_id],
        #                  import_prob_task[task_id] + 1/2 * std_values,
        #                  import_prob_task[task_id] + 1/2 * std_values,
        #                  facecolor=colors[i], alpha=0.5)
    fixed_probs = [0.3] * len(x_values)
    plt.plot(x_values, fixed_probs, 'b:', label='fixed $p_1^{tsf}$, $p_2^{tsf}$')
    plt.xlabel('$g$', font2)
    plt.ylabel('knowledge transfer probability', font2)
    y_ticks = np.arange(0, 1, 0.1)
    plt.yticks(y_ticks)
    plt.xticks(fontproperties='Times New Roman', fontsize=24)
    plt.yticks(fontproperties='Times New Roman', fontsize=24)
    plt.legend(prop={'family': 'Times New Roman', 'size': 20})
    plt.savefig(save_file, dpi=300)
    plt.close()
    print('save file to {}'.format(save_file))


def draw_from_output():
    log_file = sys.argv[1]
    task_id = int(sys.argv[2])

    update_rate_self = []
    update_rate_reuse = []
    import_prob = []
    fitness_values = []
    log_base = 10  # math.e
    selection_prob = []

    # insert rate\relative improve rate
    improve_criteria = 'relative improve rate'
    str_template = 'island {} {}'  # relative improve rate

    with open(log_file) as f:
        for line in f.readlines():
            if line.find(str_template.format(task_id, improve_criteria)) != -1:
                res = str(line.strip().split(':')[-1])
                res_list = json.loads(res)
                update_rate_self.append(float(res_list[0]))
                update_rate_reuse.append(float(res_list[1]))
                import_prob.append(float(res_list[2]))
                fitness_values.append(
                    math.log(float(res_list[3]) + 1e-12, log_base))
            if line.find(
                    'island {} selection probability'.format(task_id)) != -1:
                res = str(line.strip().split(':')[-1])
                res_list = json.loads(res)
                selection_prob.append([float(a) for a in res_list])

    # Read sto results if provided
    sto_results = []
    if len(sys.argv) == 4:
        sto_results_file = sys.argv[3]
        print('set sto results')
        with open(sto_results_file) as f:
            for line in f.readlines():
                if line.find(str_template.format(task_id,
                                                 'insert rate')) != -1:
                    res = str(line.strip().split(':')[-1])
                    res_list = json.loads(res)
                    sto_results.append(
                        math.log(float(res_list[3]) + 1e-12, log_base))

    selection_prob = np.array(selection_prob)

    gens = len(update_rate_reuse)
    show_interval = 5
    start_gens = 0
    x = list(range(gens))[start_gens::show_interval]

    plt.figure(num=0, figsize=(12, 9))
    plt.subplot(2, 2, 1)
    plt.title('Q values task {}'.format(task_id + 1))
    plt.plot(x,
             update_rate_self[start_gens::show_interval],
             'r-',
             label='self')
    plt.plot(x,
             update_rate_reuse[start_gens::show_interval],
             'b-.',
             label='reuse')
    plt.xlabel('iters')
    plt.ylabel('Q values')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.title('import prob task {}'.format(task_id + 1))
    plt.plot(x,
             import_prob[start_gens::show_interval],
             'g-.',
             label='import_prob')
    plt.xlabel('iters')
    plt.ylabel('import prob')
    # x_ticks = np.arange(0, 1, 0.1)
    y_ticks = np.arange(0, 1.1, 0.1)
    plt.yticks(y_ticks)
    plt.legend()
    plt.subplot(2, 2, 3)

    plt.title('fitness convergence task {}'.format(task_id + 1))
    plt.plot(x, fitness_values[start_gens::show_interval], 'k--', label='mto')
    # print('mto ', fitness_values[start_gens::show_interval])
    if len(sto_results) != 0:
        plt.plot(x, sto_results[start_gens::show_interval], 'b-.', label='sto')

    # print('sto ', sto_results[start_gens::show_interval])
    plt.xlabel('iters')
    plt.ylabel('fitness value (log{:.1f})'.format(log_base))
    plt.legend()

    # draw selection probabilities
    plt.subplot(2, 2, 4)
    plt.title('selection probabilities task {}'.format(task_id + 1))
    styles = [
        'r-', 'b-', 'g-', 'k-', 'r-.', 'b-.', 'g-.', 'k-.', 'r--', 'g--',
        'b--', 'k--'
    ]
    for i in range(selection_prob.shape[1]):
        plt.plot(selection_prob[:, i].T,
                 styles[i % len(styles)],
                 label='task {}'.format(i + 1))
    plt.xlabel('selection iters')
    plt.ylabel('selection probs')
    plt.yticks(y_ticks)
    plt.legend()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    save_dir = 'tmp/mtobenchmark/images'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir,
                             'update_rate_task_{}.png'.format(task_id + 1))
    plt.savefig(save_path, dpi=220)
    print('save image file to {}'.format(save_path))


if __name__ == "__main__":
    # draw_from_output()
    results_dir = 'Results/benchmark/smto/DE/deepinsight/mto/problem{}_2'

    save_dir = 'tmp/import prob/benchmark'
    # save_dir = 'tmp/import prob/mantask10/median_10'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    # results_dir = 'Results/ManyTask10/DE/deepinsight/mto/median_10'
    # results_dir = 'Results/matde_problem/DE/deepinsight/mto/zero_10'
    # for task_id in list(range(1, 11)):
    #     draw_from_results(results_dir, osp.join(save_dir,
    #                       'task_{}.pdf'.format(task_id)),
    #                       task_ids=[task_id])

    for problem_id in range(1, 10):
        problem_results = results_dir.format(problem_id)
        save_file = osp.join(save_dir, 'problem{}.pdf'.format(problem_id))
        draw_from_results(problem_results,
                          task_ids=[1, 2],
                          save_file=save_file)
