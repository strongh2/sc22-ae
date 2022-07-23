import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import matplotlib.transforms as mtransforms

colors_edge = ['#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF']
colors_bar = ['#E7D6EF', '#C399FF', '#A05CFF', '#7f66b5']
colors_cycle = ['#E7D6EF', '#C399FF', '#A05CFF', '#7f66b5']
colors_std = ['#a1a499', '#a1a499', '#a1a499', '#a1a499']
color_font = '#0F020F'

font_size = 12

_mean = lambda l: [np.mean(e) for e in l]
_std = lambda l: [np.std(e) for e in l]

def plot_2group(
    _means1, _std1, 
    _means2, _std2, 
    ylabel, xticklabels, textlabels, savefig):
    fig, ax = plt.subplots(figsize=(4, 1.5))

    width = 0.5
    ind = np.arange(len(xticklabels))

    ax.bar(ind - .4 * width,
           _means1,
           .8 * width,
           bottom=0,
           label='Baselines',
           yerr=_std1,
           color=colors_bar[0],
           edgecolor=colors_edge[0],
           ecolor=colors_std[0],
           linewidth=.6,
           error_kw={
               'elinewidth': 0.65,
               'capsize': 4
           })

    ax.bar(ind + .4 * width,
           _means2,
           .8 * width,
           bottom=0,
           label='StrongHold',
           yerr=_std2,
           color=colors_bar[2],
           edgecolor=colors_edge[2],
           ecolor=colors_std[2],
           linewidth=.6,
           error_kw={
               'elinewidth': 0.65,
               'capsize': 4
           })

    ax.set_title(None)
    ax.set_xticks(ind)
    ax.spines.top.set_visible(False)
    ax.set_xticklabels(xticklabels,
                       rotation=20,
                       ha='right',
                       fontsize=.9*font_size)

    ax.set_ylabel(ylabel, fontsize=font_size)

    ax.legend(loc='upper right',
              frameon=False,
              bbox_to_anchor=(1, 1.5),
              ncol=1,
              edgecolor=color_font,
              fontsize=.9*font_size)

    plt.savefig(savefig,
                bbox_inches='tight',
                pad_inches=0)

#----------------------------------------------------------

data, size, = {}, {}
with open('./results/case2.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        _ = row[0]+row[1]
        if row[-1] == 'B':
            if float(row[3]) >  size.get(row[2], (None, 0))[1]:
                size[row[2]] = (_, float(row[3]))
        else:
            data[_ + row[2]] = data.get(_ + row[2], [])
            data[_ + row[2]].append(float(row[4]))

d_1g_thro_base = np.array([
    data.get(size['megatron-lm'][0] + 'megatron-lm', [0]),
    data.get(size['l2l'][0] + 'l2l', [0]),
    data.get(size['zero-offload'][0] + 'zero-offload', [0]),
    data.get(size['zero-infinity'][0] + 'zero-infinity', [0]),
], dtype=object)
d_1g_thro_base_means = _mean(d_1g_thro_base)
d_1g_thro_base_std = _std(d_1g_thro_base)

d_1g_thro_ours = np.array([
    data.get(size['megatron-lm'][0] + 'stronghold', [0]),
    data.get(size['l2l'][0] + 'stronghold', [0]),
    data.get(size['zero-offload'][0] + 'stronghold', [0]),
    data.get(size['zero-infinity'][0] + 'stronghold', [0]),
], dtype=object)
d_1g_thro_ours_means = _mean(d_1g_thro_ours)
d_1g_thro_ours_std = _std(d_1g_thro_ours)

textlabels = [
    str(size['megatron-lm'][1]), 
    str(size['l2l'][1]),
    str(size['zero-offload'][1]),
    str(size['zero-infinity'][1]),
]

xticklabels = ['Megatron-LM', 'L2L', 'ZeRO-Offload', 'ZeRO-Infinity']

plot_2group(
    d_1g_thro_base_means, 
    d_1g_thro_base_std, 
    d_1g_thro_ours_means,
    d_1g_thro_ours_std, 
    'Throughput',
    xticklabels, 
    textlabels, 
    '/hostdir/metric_throughput_vs.png')
