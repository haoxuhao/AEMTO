# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import os.path as osp
import os
import re


"""
Visualize the optimum shifts of the problem shifts.
"""

data_shift_dir = "data/manytask10"
shift_degree = ["zero", "small", "median", "large"]


def draw_points(points,
                figure_indx=0,
                colors=['r', 'g', 'b'],
                marks=['s', 'D', '^']):
    plt.figure(figure_indx)
    for point in points:
        c = random.sample(colors, 1)
        mark = random.sample(marks, 1)
        plt.scatter([point[0]], [point[1]], color=c[0], marker=mark[0])
    return plt


def dim_transform(x, task_id):
    scale_factor = [200, 2, 2, 1 / 5, 1, 2]
    return scale_factor[task_id] * x


def readfile(shift_file):
    task_id = (
        int(re.search('task.*?(\d+).*?.', osp.basename(shift_file)).group(1)) -
        1) % 6
    with open(shift_file, "r") as f:
        line = f.readline()
        line = line.strip("\n")
        point = [dim_transform(float(x), task_id) for x in line.split()]

    return point


file_list = []
for tag in shift_degree:
    file_dir = osp.join(data_shift_dir, "problem5_%s" % tag)
    tmp_file_list = [osp.join(file_dir, f) for f in os.listdir(file_dir)]
    file_list.append(tmp_file_list)

figures = []
for i in range(len(shift_degree)):
    points_2dim = []
    files = file_list[i]
    for f in files:
        point = readfile(f)[:2]
        points_2dim.append(point)

    figure = draw_points(points_2dim, figure_indx=i)
    figures.append(figure)

for figure in figures:
    figure.show()
