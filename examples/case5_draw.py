import csv
import statistics
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

    ax.bar(ind, _scale, .8*width,
            bottom=0, 
            yerr=0,
            color=colors_bar[0],
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

#----------------------------------------------------------
data = {}
with open('./results/case5.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        key = int(row[0].strip().split('-')[1])
        value = float(row[9].strip())
        data[key] = data.get(key, [])
        data[key].append(value)

windows = list(data.keys())
windows.sort()
times = [statistics.median(data[k]) for k in windows]

plot_1group(
    times,
    'Iteration time (ms)', 
    windows, 
    '/hostdir/metric_window_size.png')

