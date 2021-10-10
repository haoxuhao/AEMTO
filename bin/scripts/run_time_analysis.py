import argparse
import os
import sys
import pandas as pd
import os.path as osp

from parse_results import read_tasks_results_from_json


def draw_piechart(time_sections):
    total = 0
    detailed = {}
    for k, v in time_sections.items():
        total += v
    for k, v in time_sections.items():
        print('{}: {:.2f}s ({:.2f}%)'.format(k, v, 100 * v / total))
        detailed[k] = '{:.2f}s ({:.2f}%)'.format(v, 100 * v / total)
    print('total time: {:.2f}s'.format(total))
    detailed['total'] = total
    return detailed


def get_overall_time_details(res_dir):
    res = read_tasks_results_from_json(res_dir)
    time_data = []
    for task, task_res in res.items():
        time_section = {}
        for one_run in task_res:
            for k, v in one_run['time'].items():
                if k not in time_section:
                    time_section[k] = v
                else:
                    time_section[k] += v
        time_section['task'] = task
        time_data.append(time_section)
    return pd.DataFrame(time_data)


def draw_serial_problem(res_dir):
    print('\nProblem {}'.format(osp.basename(res_dir)))
    time_pd = get_overall_time_details(res_dir)
    time_sections = dict()
    print('Run time: {:.2f}s (approximate of the job)'.format(
         (time_pd['total_time']).sum()))
    time_sections['total_time'] = time_pd['total_time'].sum()
    time_sections['total_kt_time'] = time_pd['total_kt_time'].sum()
    time_sections['problem'] = osp.basename(res_dir)
    return time_sections


def draw_problem(res_dir):
    print('\nProblem {}'.format(osp.basename(res_dir)))
    time_pd = get_overall_time_details(res_dir)
    time_sections = dict()
    print('Run time: {:.2f}s (approximate of the job)'.format(
         (time_pd['total_time'] + time_pd['wait_time']).max()))
    time_sections['transfer'] = time_pd['comm_time'].mean()
    time_sections['reuse'] = time_pd['total_kt_time'].mean()
    time_sections['EA'] = time_pd['EA_time'].mean()
    time_sections['wait_time'] = time_pd['wait_time'].mean()
    # time_sections['comm_time'] = time_pd['wait_time'].mean() + \
    #                              time_pd['comm_time'].mean()
    print('Pie chart: ')
    res = draw_piechart(time_sections)
    res['problem'] = osp.basename(res_dir)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run time analysis')
    parser.add_argument('resdir', type=str,
                        help='results dir')
    parser.add_argument('-problems', type=str,
                        help='problems, e.g., zero_100,')
    parser.add_argument('-serial', action="store_true")
    args = parser.parse_args()

    if not osp.exists(args.resdir):
        raise RuntimeError('Dir {} not exists.'.format(args.resdir))
    if args.problems:
        problems = args.problems.strip().split(',')
    else:
        files = os.listdir(args.resdir)
        problems = []
        for f in files:
            if osp.isdir(osp.join(args.resdir, f)):
                problems.append(f)

    assert len(problems) > 0 and 'problems number = 0'
    res_dir_template = osp.join(args.resdir, '{}')
    detailed_df = []
    for problem in problems:
        if args.serial:
            res = draw_serial_problem(res_dir_template.format(problem))
        else:
            res = draw_problem(res_dir_template.format(problem))
        detailed_df.append(res)
    save_dir = 'tmp/detailed_results'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    save_file = osp.join(save_dir, 'detailed_time.csv')
    pd.DataFrame(detailed_df).to_csv(save_file)
    print('save to file {}'.format(save_file))
