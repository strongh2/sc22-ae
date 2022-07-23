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

def plot_1group(_scale, ylabel, xticklabels, savefig):
    fig, ax = plt.subplots(figsize=(4, 1.5))
 
    width = 0.5  
    ind = np.arange(len(xticklabels)) 

    # draw
    colors = [colors_bar[0] for _ in range(len(xticklabels)-1)] + [colors_bar[2] ]
    ax.bar(ind, _scale, .8*width,
            bottom=0, 
            yerr=0,
            color=colors,
            edgecolor=colors_edge[0],
            ecolor=colors_std[0],
            linewidth=.6,
            error_kw={
                'elinewidth': 0.65,
                'capsize': 4})

    ax.set_title(None)
    ax.set_xticks(ind, labels=xticklabels, rotation=20, ha='right', fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.autoscale_view()

    plt.savefig(savefig, bbox_inches='tight', pad_inches=0)


xticklabels = ['Megatron-LM', 'L2L', 'ZeRO-Offload', 'ZeRO-Infinity', 'StrongHold']

data = {}
with open('./results/case1.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        key = row[0].strip()
        value = float(row[2].strip())
        data[key] = max(value, data.get(key, 0))

d_1g_scale_on_layers = [
    data.get('megatron-lm', 0), 
    data.get('l2l', 0), 
    data.get('zero-offload', 0), 
    data.get('zero-infinity', 0), 
    data.get('stronghold', 0), 
]

plot_1group(d_1g_scale_on_layers,
    'Model Size', xticklabels, '/hostdir/metric_model_scale.png')
