import csv
import statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import matplotlib.transforms as mtransforms

colors_edge = ['#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF']
colors_bar = ['#E7D6EF', '#C399FF', '#A05CFF', '#7A69b0']
colors_cycle = ['#E7D6EF', '#C399FF', '#A05CFF', '#7A69b0']
colors_std = ['#a1a499', '#a1a499', '#a1a499', '#a1a499']
color_font = '#0F020F'

font_size = 10

_mean = lambda l: [np.mean(e) for e in l]
_std = lambda l: [np.std(e) for e in l]

def plot(_data, _indices, savefig):
    fig, ax = plt.subplots(figsize=(4, 1.5))

    width = 0.5

    ax.plot(_indices, _data, color=colors_bar[2], label='StrongHold')

    ax.set_xticks(_indices)

    ax.set_ylabel('Iteration Time (ms)', fontsize=.9 * font_size)
    ax.set_xlabel('#Layers', fontsize=font_size)

    ax.legend(loc='upper left', frameon=False, edgecolor=color_font)

    plt.savefig(savefig,
                bbox_inches='tight',
                pad_inches=0)


#----------------------------------------------------------
data = {}
with open('./results/case4.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        key = int(row[0].strip().split('-')[1])
        value = float(row[11].strip())
        data[key] = data.get(key, [])
        data[key].append(value)

layers = list(data.keys())
layers.sort()
times = [statistics.median(data[k]) for k in layers]

plot(times, layers, '/hostdir/metric_linear_scaling.png')
