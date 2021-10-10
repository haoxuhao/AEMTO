import os

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp


def draw_multi_lines(x_data, y_data, labels,
                     x_label=None, y_label=None,
                     save_file='tmp/params_analysis/tmp.pdf'):
    """draw multiple lines
    x_data (numpy): parameter settings
    y_data (numpy): values under different parameter settings
    labels (list): the labels of each y_data
    """
    assert x_data.shape[0] == y_data.shape[1]
    assert y_data.shape[0] == len(labels)
    if not osp.exists(osp.dirname(save_file)):
        os.makedirs(osp.dirname(save_file))
    N = y_data.shape[0]
    styles = ['k-.', 'k-.', 'k-.', 'k-.', 'k-.',
              'k-', 'k-', 'k-', 'k-', 'k-']
    styles = ['r-.', 'k-', 'b--', 'g:']

    markers = ["D", "x", "o", ">", "s", ".", "D"]
    font2 = {
                'family': 'Times New Roman',
                'weight': 'normal',
                'size': 28,
    }
    # plt.figure(figsize=(8, 6.5))
    fig = plt.figure(figsize=(8, 6.5))
    # left, bottom, width, height (range 0 to 1)
    left, bottom = 0.15, 0.15
    ax = fig.add_axes([left, bottom, (1 - left) * 0.9, (1 - bottom) * 0.9])
    for i in range(N):
        y = y_data[i, :]
        plt.plot(x_data, y, styles[i % N], marker=markers[i], label=labels[i])
    plt.legend(loc=0, prop={'family': 'Times New Roman', 'size': 20})
    if x_label is not None:
        plt.xlabel(x_label, fontdict=font2)
    if y_label is not None:
        plt.ylabel(y_label, fontdict=font2)
    plt.xticks(fontproperties='Times New Roman', fontsize=24)
    plt.yticks(fontproperties='Times New Roman', fontsize=24)
    # plt.tight_layout()
    plt.savefig(save_file, dpi=220)
    print('save file to {}'.format(save_file))
    plt.close()


if __name__ == '__main__':
    """upper bound import probability analysis"""
    x_data = np.array([0.3, 0.5, 0.7, 0.8, 0.9, 0.95])
    y_data = np.array([
        np.array([30, 26.50, 20.50, 27.50, 42.00, 42.50]) / 9.0,
        (10 * np.array([4.7, 3.40, 2.20, 2.80, 3.50, 4.40]) +
            10 * np.array([19.9, 15.60, 12.60, 11.5, 10.90, 13.50])) / 50,
        np.array([27.14, 23.24, 20.78, 19.00, 17.42, 18.42]) / 6.0
    ])
    labels = ['Test suite 1', 'Test suite 2', 'Test suite 3']
    draw_multi_lines(x_data, y_data,
                     labels=labels,
                     x_label='$p_{ub}^{tsf}$',
                     y_label='ranking',
                     save_file='tmp/params_analysis/upper_import_prob.pdf')

    """pbase analysis"""
    x_data = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
    y_data = np.array([
        (10 * np.array([1.9, 2.8, 2.6, 4.10, 4.4, 5.58, 6.40]) +
            np.array([7.6, 8.6, 13.60, 14.70, 19.40, 21.30, 26.80]) * 10) / 50,
        np.array([23, 22.94, 24.52, 23.96, 21.26, 27.50, 24.80]) / 6.0,
        np.array([17.70, 13.70, 12.70, 14.20, 15.60, 17.50, 20.60]) / 4.0,
    ])

    labels = ['Test suite 2', 'Test suite 3',
              'ManyTask10-* (100D)']
    draw_multi_lines(x_data, y_data,
                     labels=labels,
                     x_label='$p_{base}$',
                     y_label='ranking',
                     save_file='tmp/params_analysis'
                               '/upper_pbase.pdf')
