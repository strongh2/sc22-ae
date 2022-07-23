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

def plot_1group(_means, _std, ylabel, xticklabels, savefig):
    fig, ax = plt.subplots(figsize=(4, 1.5))
 
    width = 0.5
    ind = np.arange(len(xticklabels)) 

    colors = [ colors_bar[0] for i in range(len(xticklabels)-1)] + [colors_bar[1] ]

    ax.bar(ind, _means, .8*width, 
            bottom=0, 
            yerr=_std,
            color=colors,
            edgecolor=colors_edge[0],
            ecolor=colors_std[0],
            linewidth=.6,
            error_kw={
                'elinewidth': 0.65,
                'capsize': 4})

    ax.set_title(None)
    ax.set_xticks(ind)
    ax.set_xticklabels(xticklabels, rotation=20, ha='right',  fontsize=.9*font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)

    ax.autoscale_view()

    plt.savefig(savefig, bbox_inches='tight')

#----------------------------------------------------------
data = {}
with open('./results/case3.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        key = row[2].strip()
        value = float(row[4].strip())
        data[key] = data.get(key, [])
        data[key].append(value)

d_1g_thro = np.array([
    data.get('megatron-lm', [0]), 
    data.get('l2l', [0]), 
    data.get('zero-offload', [0]), 
    data.get('zero-infinity', [0]), 
    data.get('stronghold', [0]),
    ], dtype=object)
d_1g_thro_means = _mean(d_1g_thro)
d_1g_thro_std = _std(d_1g_thro)

xticklabels = ['Megatron-LM', 'L2L', 'ZeRO-Offload', 'ZeRO-Infinity',  'StrongHold']

plot_1group(
    d_1g_thro_means, 
    d_1g_thro_std, 
    'Throughput',
     xticklabels, 
     '/hostdir/metric_throughput.png')

