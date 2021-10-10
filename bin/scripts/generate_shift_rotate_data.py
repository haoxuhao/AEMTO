# encoding=utf-8
import random
import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from scipy.stats import special_ortho_group
from scipy.io import loadmat
from tqdm import tqdm
from sklearn.cluster import KMeans


def flush_data_to_file(data, file):
    with open(file, "w") as f:
        for line in data:
            for e in line:
                f.write(str(e))
                f.write(" ")
            f.write("\n")


def generate_uniform_shift_data(shift_range=[-1, 1],
                                origin_space=[-100, 100],
                                origin_opt=0.0,
                                unified_sapce=[0, 1],
                                shift_opt=None,
                                dim=100,
                                return_origin=True):
    ret = [0] * dim
    scale_factor = (origin_space[1] - origin_space[0]) / \
                   (unified_sapce[1] - unified_sapce[0])

    for i in range(dim):
        if isinstance(shift_opt, list):
            ret[i] = ((random.uniform(shift_range[0], shift_range[1])) -
                      shift_opt[i]) * scale_factor - origin_opt
        else:
            ret[i] = ((random.uniform(shift_range[0], shift_range[1])) -
                      shift_opt) * scale_factor - origin_opt
    return ret


def generate_shift_data_10_tasks():
    save_path = "./data/manytask_shiftdata_10_tasks_100dim"
    if not osp.exists(save_path):
        os.makedirs(save_path)

    max_dim = 100

    data = [[0] * max_dim]
    filename = osp.join(save_path, "shift_task1.txt")
    flush_data_to_file(data, filename)

    data = [[80] * max_dim]
    filename = osp.join(save_path, "shift_task2.txt")
    flush_data_to_file(data, filename)

    data = [[-80] * max_dim]
    filename = osp.join(save_path, "shift_task3.txt")
    flush_data_to_file(data, filename)

    data = [[-0.4] * (max_dim // 2)]
    filename = osp.join(save_path, "shift_task4.txt")
    flush_data_to_file(data, filename)

    data = [[0] * max_dim]
    filename = osp.join(save_path,
                        "shift_task5.txt")  # shift task5's global optimum
    flush_data_to_file(data, filename)

    data = [[40] * max_dim]
    filename = osp.join(save_path, "shift_task6.txt")
    flush_data_to_file(data, filename)

    data = [[-0.4] * max_dim]
    filename = osp.join(save_path, "shift_task7.txt")
    flush_data_to_file(data, filename)

    # data = [[420.9687]*50]
    data = [[420.9687] * max_dim]
    filename = osp.join(save_path, "shift_task8.txt")
    flush_data_to_file(data, filename)

    data = [[-80] * (max_dim // 2) + [80] * (max_dim // 2)]
    filename = osp.join(save_path, "shift_task9.txt")
    flush_data_to_file(data, filename)

    data = [[40] * (max_dim // 2) + [-40] * (max_dim // 2)]
    filename = osp.join(save_path, "shift_task10.txt")
    flush_data_to_file(data, filename)


def generate_benchmark_9problems(data_root="./data/mto_benchmark"):
    '''
    this function generate the 9 Multi-task benchmark problems
    '''
    problem_settings = [
        "CI_H", "CI_M", "CI_L", "PI_H", "PI_M", "PI_L", "NI_H", "NI_M", "NI_L"
    ]
    rotation_matrix_prefix = "Rotation_Task"
    shift_vector_prefix = "GO_Task"

    for i, s in tqdm(enumerate(problem_settings)):
        problem_id = i + 1
        mat_file = osp.join(data_root, s + ".mat")
        mat = loadmat(mat_file)
        problem_save_dir = osp.join(data_root, "problem%d" % problem_id)
        if not osp.exists(problem_save_dir):
            os.makedirs(problem_save_dir)

        for t in range(1, 3):
            name = rotation_matrix_prefix + str(t)
            if name in mat:
                rotation_txt_file = osp.join(problem_save_dir,
                                             "rotation_%d.txt" % t)
                data = mat[name].tolist()
                flush_data_to_file(data, rotation_txt_file)
            name = shift_vector_prefix + str(t)
            if name in mat:
                shift_txt_file = osp.join(problem_save_dir, "shift_%d.txt" % t)
                data = mat[name].tolist()
                flush_data_to_file(data, shift_txt_file)
        if problem_id == 3:
            shift_txt_file = osp.join(problem_save_dir, "shift_%d.txt" % 2)
            shift_data = [[0] * 50]
            flush_data_to_file(shift_data, shift_txt_file)
        if problem_id == 5:
            shift_txt_file = osp.join(problem_save_dir, "shift_%d.txt" % 2)
            shift_data = [[0] * 50]
            flush_data_to_file(shift_data, shift_txt_file)
        if problem_id == 7:
            shift_txt_file = osp.join(problem_save_dir, "shift_%d.txt" % 1)
            shift_data = [[0] * 50]
            flush_data_to_file(shift_data, shift_txt_file)
        if problem_id == 9:
            shift_txt_file = osp.join(problem_save_dir, "shift_%d.txt" % 2)
            shift_data = [[0] * 50]
            flush_data_to_file(shift_data, shift_txt_file)


def generate_shift_rotation_data_manytask_10(
        problem_id,
        data_root="./data/CEC50",
        var_range=[-50, 50]):
    shift_data_problem_file = osp.join(data_root, "GoTask%d.mat" % problem_id)
    rotation_data_problem_file = osp.join(data_root,
                                          "RotationTask%d.mat" % problem_id)

    save_root = osp.join(data_root, "problem%d" % problem_id)
    if not osp.exists(save_root):
        os.makedirs(save_root)

    shift_data_problem = loadmat(shift_data_problem_file)["GoTask%d" %
                                                          problem_id]

    shift_data_problem_unified_space = (shift_data_problem + var_range[0]) / (
        var_range[1] - var_range[0])  # [0,1]
    print(shift_data_problem.shape)
    # if problem_id == 6:
    #     shift_data_problem -= 420.9687
    # if problem_id == 1:
    #     shift_data_problem -= 1

    print(np.var(shift_data_problem_unified_space, axis=1))

    rotation_data_problem = loadmat(rotation_data_problem_file)[
        "RotationTask%d" % problem_id][0]
    print(rotation_data_problem[1].shape)

    total_component_tasks = 50
    dim = 50

    for i in tqdm(range(total_component_tasks)):
        # save the shift data
        shift_save_path = osp.join(save_root, "shift_%d.txt" % (i + 1))
        with open(shift_save_path, "w") as f:
            for j in range(dim):
                f.write("%f " % shift_data_problem[i, j])
            f.write("\n")

        # save the rotation data
        rotation_save_path = osp.join(save_root, "rotation_%d.txt" % (i + 1))
        tmp_rotation_data = rotation_data_problem[i]

        with open(rotation_save_path, "w") as f:
            for j in range(dim):
                for k in range(dim):
                    f.write("%f " % tmp_rotation_data[j, k])
                f.write("\n")

    print("done.")


def generate_arm_tasks():
    clusters = 100
    centers_of_cluster = 20
    D = 2
    save_root = 'data/Arm/uniform_centers/'
    if not osp.exists(save_root):
        os.makedirs(save_root)

    print('generate {} random centers in {} dimension '
          'space using kmeans'.format(clusters, D))
    x = np.random.rand(clusters * 20, D)
    k_means = KMeans(init='k-means++', n_clusters=clusters,
                     n_init=1, n_jobs=-1, verbose=1)
    k_means.fit(x)
    rand_centers = k_means.cluster_centers_
    shifts = [0, 0.005, 0.025, 0.05]
    tags = ['zero', 'small', 'median', 'large']
    problem_savedir = osp.join(
        save_root, 'cluster{}_center{}_D{}'.format(
            clusters,
            centers_of_cluster,
            D,
        ))
    if not osp.exists(problem_savedir):
        os.makedirs(problem_savedir)
    for i, tag in enumerate(tags):
        srange = shifts[i]
        print('problem: {}, {} shifts'.format(problem_savedir, tag))
        task_cnt = 0
        data_all = []
        for j in tqdm(range(rand_centers.shape[0])):
            centers = [rand_centers[j, k] for k in range(D)]
            for _ in range(centers_of_cluster):
                shift_data = [
                    c + random.uniform(-srange, srange)
                    for c in centers
                ]
                for k in range(len(shift_data)):
                    if shift_data[k] < 1e-2:
                        shift_data[k] = 1e-2
                    if shift_data[k] > 1 - 1e-2:
                        shift_data[k] = 1 - 1e-2

                task_cnt += 1
                data_all.append(shift_data)
        all_file = osp.join(problem_savedir, "{}_shift.dat".format(tag))
        flush_data_to_file(data_all, all_file)
        print('tag {}, ntasks {}'.format(tag, task_cnt))


def generate_manymany_tasks():
    base_functions = [
        {
            'name': 'rosenbrock',
            'range': [-50, 50],
            'opt': 1,
        },
        {
            'name': 'ackley',
            'range':  [-50, 50],
            'opt': 0.0
        },
        {
            'name': 'schwefel',
            'range':  [-500, 500],
            'opt': 420.9687,
        },
        {
            'name': 'griewank',
            'range':  [-100, 100],
            'opt': 0,
        },
        {
            'name': 'rastrgin',
            'range': [-50, 50],
            'opt': 0,
        },
    ]

    # total tasks = clusters * centers_of_cluster
    clusters = 100
    centers_of_cluster = 20
    D = 50
    save_root = 'data/ManMany/uniform_centers/'

    print('generate {} random centers in {} dimension '
          'space using kmeans'.format(clusters, D))
    x = np.random.rand(clusters * 20, D)
    k_means = KMeans(init='k-means++', n_clusters=clusters,
                     n_init=1, n_jobs=-1, verbose=1)
    k_means.fit(x)
    rand_centers = k_means.cluster_centers_
    shifts = [0, 0.005, 0.025, 0.05]
    tags = ['zero', 'small', 'median', 'large']
    problem_savedir = osp.join(
        save_root, 'cluster{}_center{}_D{}'.format(
            clusters,
            centers_of_cluster,
            D,
        ))
    if not osp.exists(problem_savedir):
        os.makedirs(problem_savedir)

    for i, tag in enumerate(tags):
        srange = shifts[i]
        task_cnt = 0
        data_all = []
        for j in tqdm(range(rand_centers.shape[0])):
            centers = [rand_centers[j, k] - 0.5 for k in range(D)]
            for _ in range(centers_of_cluster):
                shift_data = generate_uniform_shift_data(
                    shift_range=[-srange, srange],
                    origin_space=base_functions[task_cnt %
                                                len(base_functions)]['range'],
                    origin_opt=base_functions[task_cnt %
                                              len(base_functions)]['opt'],
                    shift_opt=centers,
                    dim=D)
                task_cnt += 1
                data_all.append(shift_data)
        all_file = osp.join(problem_savedir, "{}_shift.dat".format(tag))
        flush_data_to_file(data_all, all_file)
        print('problem {}, ntasks {}'.format(problem_savedir, task_cnt))


def generate_my_manytask_problems():
    base_functions = [
        # {
        #     'name': 'weirstrass',
        #     'range': [-0.5, 0.5],
        #     'opt': 0.0
        # },
        # {
        #     'name': 'rosenbrock',
        #     'range': [-50, 50],
        #     'opt': 1,
        # },
        # {
        #     'name': 'ackley',
        #     'range':  [-50, 50],
        #     'opt': 0.0
        # },
        # {
        #     'name': 'schwefel',
        #     'range':  [-500, 500],
        #     'opt': 420.9687,
        # },
        # {
        #     'name': 'griewank',
        #     'range':  [-100, 100],
        #     'opt': 0,
        # },
        # {
        #     'name': 'rastrgin',
        #     'range': [-50, 50],
        #     'opt': 0,
        # }
        {
            'name': 'cf',
            'range':  [-100, 100],
            'opt': 0,
        },
        {
            'name': 'cf',
            'range':  [-100, 100],
            'opt': 0,
        },
        {
            'name': 'cf',
            'range':  [-100, 100],
            'opt': 0,
        },
        {
            'name': 'cf',
            'range':  [-100, 100],
            'opt': 0,
        },
        {
            'name': 'cf',
            'range':  [-100, 100],
            'opt': 0,
        },
    ]
    # [0.25, 0.375, 0.5, 0.625, 0.75]
    # global_optimumns = [0/200.0, -25/200.0, 25/200.0, -50/200.0, 50/200.0]
    global_optimumns = [-50/200.0, 50/200.0]
    shift_tags = ["zero", "small", "median", "large", "llarge"]
    shift_range = {}
    shift_range["zero"] = [0, 0]  # 0
    shift_range["small"] = [-1/200.0, 1/200.0]  # 0.005
    shift_range["median"] = [-5/200.0, 5/200.0]  # 0.025
    shift_range["large"] = [-10/200.0, 10/200.0]  # 0.05
    shift_range["llarge"] = [-15/200.0, 15/200.0]  # 0.075

    save_root = "./data/manytask_complicated"
    if not osp.exists(save_root):
        os.makedirs(save_root)

    problem_id = 5
    for tag in shift_tags:
        task_id = 1
        for g in range(2):
            problem_dir = osp.join(save_root,
                                   "problem%d_%s" % (problem_id, tag))
            if not osp.exists(problem_dir):
                os.makedirs(problem_dir)
            for func in base_functions:
                print(func['name'], func['range'], func['opt'])
                shift_data = generate_uniform_shift_data(
                                shift_range=shift_range[tag],
                                origin_space=func['range'],
                                origin_opt=func['opt'],
                                shift_opt=global_optimumns[g],
                                dim=50)
                task_shift_file = osp.join(problem_dir,
                                           "shift_task%d.txt" % (task_id))
                flush_data_to_file([shift_data], task_shift_file)

                # task_rotate_file = osp.join(problem_dir,
                #                             "rotate_task%d.txt" % (task_id))
                # flush_data_to_file(special_ortho_group.rvs(50).tolist(),
                #                    task_rotate_file)
                task_id += 1


def shifts_vis(save_path='tmp/shift_vis.pdf'):
    global_optimumns = [0.25, 0.75]
    shift_range = {}
    shift_range["zero"] = [0, 0]  # 0
    shift_range["small"] = [-1/200.0, 1/200.0]  # 0.005
    shift_range["median"] = [-5/200.0, 5/200.0]  # 0.025
    shift_range["large"] = [-10/200.0, 10/200.0]  # 0.05
    shift_range["llarge"] = [-15/200.0, 15/200.0]  # 0.07
    shift_tags = ["median", "large"]
    vis_dim = 2
    ntasks = 5
    markers = ['.', 'x']
    font2 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 18,
    }
    plt.figure()
    for i, tag in enumerate(shift_tags):
        for j, opt in enumerate(global_optimumns):
            shift_low, shift_high = shift_range[tag]
            points = np.random.uniform(shift_low, shift_high,
                                       (vis_dim, ntasks)) + opt
            if j == 0:
                plt.scatter(points[0, :], points[1, :], c='k',
                            marker=markers[i % len(markers)], label=tag)
            else:
                plt.scatter(points[0, :], points[1, :], c='k',
                            marker=markers[i % len(markers)])
    # plot two centers with red '+'
    plt.scatter(global_optimumns[0], global_optimumns[0], c='r', marker='+',
                label='center')
    plt.scatter(global_optimumns[1], global_optimumns[1], c='r', marker='+')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('$x_1$', fontdict=font2)
    plt.ylabel('$x_2$', fontdict=font2)
    plt.legend(loc=1, prop={'family': 'Times New Roman', 'size': 12})
    plt.savefig(save_path, dpi=300)
    print('save image file to {}'.format(save_path))
    plt.close()


if __name__ == "__main__":
    random.seed(666)
    # generate_benchmark_9problems()
    # generate_my_manytask_problems()
    generate_manymany_tasks()
    # generate_arm_tasks()
    # shifts_vis()
    # M = special_ortho_group.rvs(3)
    # print(M.tolist())
    # # generate_shift_data_10_tasks()
    # range_data = [[-50, 50], [-50, 50], [-50, 50], [-100, 100], [-0.5, 0.5],
    #               [-500, 500]]
    # range_data = [[-100, 100], [-100, 100], [-100, 100],
    #   [-100, 100], [-100, 100]]
    # for i in range(6, 7):
    #     problem_id = i
    #     generate_shift_rotation_data_manytask_10(
    #         problem_id, var_range=range_data[problem_id - 1])
    #     break
