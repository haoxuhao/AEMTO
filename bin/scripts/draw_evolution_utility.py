# coding=utf8
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import os

from parse_results import read_tasks_results_from_json


matplotlib.rcParams['axes.unicode_minus'] = False


def draw_utility_task(results, task_id,
                      labels=["1", "2", "3", "4", "5"],
                      markers=["D", "x", "o", ">", "s", ".", "D"],
                      figure_index=0,
                      interval=50,
                      utility_tag='Accumulated population\n member update rate',
                      yname=None,
                      with_marks=True,
                      save_file='./utility_visualization/utility.png'):
    """plot the mean utility values over different algorithms
    """
    styles = ['r-.', 'k-', 'b--', 'g:']
    fontdict = {'family': 'Times New Roman', 'size': 28}
    fig = plt.figure(figsize=(8, 6.5))
    # left, bottom, width, height (range 0 to 1)
    left, bottom = 0.225, 0.15
    ax = fig.add_axes([left, bottom, (1 - left) * 0.94, (1 - bottom) * 0.9])
    for j, res in enumerate(results):
        marker = markers[j]
        utility_values = [obj[utility_tag] for obj in res[task_id]]
        averaged_utility = np.mean(utility_values, axis=0)[1:]
        generations = list(range(1, averaged_utility.shape[0] + 1))
        averaged_utility = \
            averaged_utility * np.array(generations) / generations[-1]
        plt.plot(generations[0::interval], averaged_utility[0::interval],
                 styles[j], label=labels[j],
                 marker=marker)

    plt.legend(loc=0, prop={'family': 'Times New Roman', 'size': 20})
    plt.xlabel('$g$', fontdict=fontdict)
    if yname is None:
        plt.ylabel(utility_tag, fontdict=fontdict)
    else:
        plt.ylabel(yname, fontdict={'family': 'Times New Roman', 'size': 24})
    # plt.ylim([0, 0.5])
    plt.xticks(fontproperties='Times New Roman', fontsize=24)
    plt.yticks(fontproperties='Times New Roman', fontsize=24)
    # plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()
    print('save image file to {}'.format(save_file))


def process_one_problem(
        results, labels, save_dir, task_ids,
        utility_tag='success_best_update_rate', yname=None):
    assert len(results) == len(labels), 'results len == labels len'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    for task_id in task_ids:
        draw_utility_task(
            results, task_id, labels=labels, utility_tag=utility_tag,
            yname=yname,
            save_file=osp.join(save_dir, 'task{}.pdf'.format(task_id)))


if __name__ == '__main__':
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
    # results_dirs = [
    #     'Results/benchmark/smto/DE/deepinsight/mto/{}',
    #     'Results/benchmark/smto/DE/deepinsight/mto_fixed_tsf/{}',
    #     'Results/benchmark/smto/DE/deepinsight/sto/{}',
    # ]
    # labels = ['AEMTO', 'AEMTO (w/o aTsf)', 'STO']
    # task_ids = range(1, 3)
    # problems = ['problem{}_2'.format(id) for id in range(1, 10)]
    # problem_save_root = 'mtobenchmark'

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

    utility_tags = {
        # 'success_best_update_rate': 'best individual update rate',
        'success_offsprings_rate': 'accumulated population\nmember update rate'
    }
    save_dir_prefix = 'tmp/utility_visualization/'\
        '{}/{}/{}'
    for problem_tag in problems:
        print('process problem {}'.format(problem_tag))
        results = [
            read_tasks_results_from_json(
                res_dir.format(problem_tag)) for res_dir in results_dirs
        ]
        for utility_tag, name in utility_tags.items():
            save_dir = save_dir_prefix.format(
                problem_save_root, utility_tag, problem_tag)
            process_one_problem(results, labels, save_dir,
                                task_ids, yname=name, utility_tag=utility_tag)
