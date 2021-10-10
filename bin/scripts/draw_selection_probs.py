import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from parse_results import read_tasks_results_from_json


def draw_transfer_matrix(task_selection_probs_mean,
                         save_file='tmp/transfer_matrix/transfer_matrix.pdf',
                         label_display_interval=5,
                         with_text=False,
                         auto_scale=False):
    # draw transfer matrixs
    ntasks = task_selection_probs_mean.shape[0]
    lables = [str(i + 1) for i in range(ntasks)]
    plt.figure(num=0)
    if not auto_scale:
        plt.imshow(
            task_selection_probs_mean,
            vmin=0,  cmap='gray_r',
            vmax=1)
    else:
        plt.imshow(task_selection_probs_mean, cmap='gray_r')
    # plt.colorbar()
    cbar = plt.colorbar()
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(20)
    tick_marks = np.arange(len(lables))
    plt.xticks(tick_marks, lables, fontproperties='Times New Roman', fontsize=20)
    plt.yticks(tick_marks, lables, fontproperties='Times New Roman', fontsize=20)

    # add text
    if with_text:
        w, h = task_selection_probs_mean.shape[1],\
               task_selection_probs_mean.shape[0]
        iters = np.reshape([[[i, j] for j in range(w)] for i in range(h)],
                           (task_selection_probs_mean.size, 2))
        for i, j in iters:
            plt.text(
                j - 0.4,
                i,
                ("%.2f" % task_selection_probs_mean[i, j]),
                size=10,
                color="k",
                weight="light",
            )
    font2 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 24,
    }

    plt.ylabel('target task', font2)
    plt.xlabel('source task', font2)
    plt.tight_layout()
    ax = plt.gca()
    ntasks = task_selection_probs_mean.shape[1]

    x_ticks = list(
        range(label_display_interval - 1, ntasks, label_display_interval))
    if label_display_interval != 1:
        x_ticks = [0] + x_ticks
    ax.set_xticks(x_ticks)
    ax.set_yticks(x_ticks)

    x_labels = list(
        range(label_display_interval, ntasks + 1, label_display_interval))

    if label_display_interval != 1:
        x_labels = [1] + x_labels
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(x_labels)

    plt.savefig(save_file, dpi=300)
    print('save image file to {}'.format(save_file))


def draw_import_prob_trend(task_selection_probs_np,
                           skip_id=0,
                           import_prob_mean=None,
                           import_prob_std=None,
                           interval=50,
                           save_file='tmp/selection_probs/task_0.pdf'):
    mean_over_multi_runs = np.mean(task_selection_probs_np, axis=0)
    styles = ['k-.', 'k-.', 'k-.', 'k-.', 'k-.', 'k-', 'k-', 'k-', 'k-', 'k-']
    markers = ["*", "x", "o", "s", ">", "*", "x", "o", "s", ">"]
    font2 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 24,
    }
    if import_prob_mean is not None and import_prob_std is not None:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for i in range(mean_over_multi_runs.shape[1]):
            if i != skip_id:
                prob_trend_data = mean_over_multi_runs[:, i].tolist()
                generations = list(range(0, len(prob_trend_data)*interval))
                generations = generations[0::interval]
                prob_trend_data = prob_trend_data
                ax1.plot(generations,
                         prob_trend_data,
                         styles[i],
                         label=r'$\tau_{%d}$' % (i + 1),
                         marker=markers[i])
        # ax1.set_title("$T_{%s}$" % str(skip_id+1), fontdict=font2)
        ax1.set_ylabel(r'selection probability',
                       font2)
        ax1.set_ylim([0, 1.1])
        ax1.set_xlabel('$g$', font2)
        ax1.legend(loc=2, prop={'family': 'Times New Roman', 'size': 20},
                   ncol=3)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.tick_params(axis='both', which='minor', labelsize=20)
        
        # ax2 = ax1.twinx()
        # ax2.set_ylim([0, 1])
        # ax2.set_ylabel('import probability ($p^{im}$)', font2)
        # ax2.set_xlabel('$g$', font2)
        # generations = list(range(0, len(import_prob_mean)))
        # generations = generations[0::interval]
        # ax2.errorbar(generations, import_prob_mean[0::interval],
        #              yerr=import_prob_std[0::interval], fmt='o-', ecolor='r',
        #              color='b', elinewidth=2, capsize=4,
        #              label='$p^{im}$',
        #              marker='D')
        # ax2.legend(loc=1, prop={'family': 'Times New Roman', 'size': 10})
    else:
        plt.figure()
        plt.title("$T_{%s}$" % str(skip_id + 1), fontdict=font2)
        for i in range(mean_over_multi_runs.shape[1]):
            if i != skip_id:
                prob_trend_data = mean_over_multi_runs[:, i].tolist()
                prob_trend_data = prob_trend_data[0::interval]
                plt.plot(generations,
                         prob_trend_data,
                         styles[i],
                         label=r'$\tau_{%d}$' % (i + 1),
                         marker=markers[i])
        plt.legend(loc=0,
                   prop={'family': 'Times New Roman', 'size': 20}, ncol=3)
        plt.xlabel('$g$', font2)
        plt.ylabel(r'selection probability' % (skip_id + 1), font2)
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)

    plt.close()
    print('save to file {}'.format(save_file))


def draw_from_record_results(results_file,
                             task_ids=list(range(1, 51)),
                             save_dir='tmp/selection probs/'
                             'matde_with_importprob'):
    """Read the recorded selection probs from the results file"""
    print('loading data from {}'.format(results_file))
    results = read_tasks_results_from_json(results_file)
    runs = len(results[task_ids[0]])
    print('runs', runs)
    task_selection_probs_mean = []
    task_selection_trend = {}
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    for i, task_id in enumerate(task_ids):
        task_selection_probs = []
        task_import_probs = []
        for run_id in range(runs):
            selection_probs = results[task_id][run_id]['selection_probs']
            task_import_probs.append(results[task_id][run_id]['import_prob'])
            task_selection_probs.append(selection_probs)
        task_import_probs_np = np.array(task_import_probs)
        task_selection_probs_np = np.array(task_selection_probs)
        mean_values = np.mean(task_import_probs_np, axis=0)
        std_values = np.std(task_import_probs_np, axis=0)

        mean_selections = np.mean(task_selection_probs_np,
                                  axis=(0, 1)).tolist()
        task_selection_probs_mean.append(mean_selections)
        task_selection_trend[task_id] = task_selection_probs_np
        save_file = osp.join(save_dir,
                             'selection_probs_task_{}.pdf'.format(task_id))
        draw_import_prob_trend(task_selection_trend[task_id],
                               import_prob_mean=mean_values,
                               import_prob_std=std_values,
                               skip_id=i, save_file=save_file)

    task_selection_probs_mean = np.array(task_selection_probs_mean)
    draw_transfer_matrix(task_selection_probs_mean,
                         save_file=osp.join(save_dir, 'transfer_matrix.pdf'))


if __name__ == '__main__':
    ntasks = 10
    tag = 'median_{}'.format(ntasks)
    problem = 'ManyTask10'
    results_file = 'Results/{}/smto/DE/'\
                   'deepinsight/mto/{}'.format(problem, tag)

    draw_from_record_results(results_file,
                             task_ids=range(1, ntasks + 1),
                             save_dir='tmp/selection probs/'
                             '{}/{}'.format(problem, tag))
