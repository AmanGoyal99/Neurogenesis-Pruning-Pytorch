import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.transforms import ScaledTranslation
from statsmodels.stats.weightstats import ztest

from src.utils.utils import load_cfg, load_logs


# source: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() .rotate_deg(45) .scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def savefig(fig, filename):
    fig.savefig(f'figs/{filename}.pdf', bbox_inches='tight', dpi=200)


def mean_and_std(ax, run_list, neuron_list, color_idx=0, scale=1., label='', w_std=True, reference=None, logx=False):
    color = f'C{color_idx}'
    r = np.vstack(run_list) / np.vstack(neuron_list) 
    m = np.mean(r, axis=0)
    if ax is not None:
        x = np.arange(len(m))
        if logx:
            x = np.log10(x+1)
        ax.plot(x, m, color=color, label=label)
        if len(run_list) > 1:
            s = np.std(r, axis=0)
            ax.fill_between(x, m + s, m - s, alpha=0.2, color=color)


def multi_rank_plot(ax, name, sample_no=None, metric='erank'):
    r_dict = load_logs(name)
    cfg = load_cfg(name)

    def r_value(key):
        if sample_no is None:
            return r_dict[key]
        else:
            return [r_dict[key][sample_no]]

    def plot_layer(layer, color):
        r = r_value(f'{layer}_{metric}')
        c = r_value(f'{layer}_out_channel')
        mean_and_std(ax, r, c, color, label=layer)

    for i in range(cfg['n_conv_layers']):
        plot_layer(f'conv{i}', i)
    for i in range(cfg['n_fc_layers']):
        plot_layer(f'fc{i}', i + cfg['n_conv_layers'])

    ax.legend()


def multi_simple_plot(ax, name, attr, sample_no=None):
    r_dict = load_logs(name)
    cfg = load_cfg(name)

    def r_value(key):
        if sample_no is None:
            return r_dict[key]
        else:
            return [r_dict[key][sample_no]]

    def plot_layer(layer, color):
        v = r_value(f'{layer}_{attr}')
        mean_and_std(ax, v, [[1.]], color, label=layer)

    for i in range(cfg['n_conv_layers']):
        plot_layer(f'conv{i}', i)
    for i in range(cfg['n_fc_layers']):
        plot_layer(f'fc{i}', i + cfg['n_conv_layers'])

    ax.legend()


def erank_vs_edim():
    name = 'baseline'
    sample_no = None
    width, height = 6, 4
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)

    labels = [r'$\epsilon$-numerical rank', 'erank']
    attrs = ['edim', 'erank']
    colors = [5, 0]

    for i in range(2):
        r_dict = load_logs(name)
        cfg = load_cfg(name)

        def r_value(key):
            if sample_no is None:
                return r_dict[key]
            else:
                return [r_dict[key][sample_no]]

        def plot_layer(layer, color):
            v = r_value(f'{layer}_{attrs[i]}')
            mean_and_std(ax, v, [[1.]], color, label=labels[i])

        plot_layer(f'conv0', colors[i])

    ax.set_ylim([0, 17])
    ax.set_ylabel('rank metric')
    ax.set_xlabel('steps')
    ax.legend(loc='lower right')

    savefig(fig, 'erank_vs_edim')
    plt.show()

def baseline_erank():
    name = 'baseline'
    width, height = 6, 4
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)

    multi_simple_plot(ax, name, 'erank', sample_no=None)

    ax.set_ylim([0, 60])
    ax.set_ylabel('erank')
    ax.set_xlabel('steps')
    ax.legend(loc='lower right')

    savefig(fig, 'baseline_erank')
    plt.show()

def growing():
    width, height = 4, 4

    names = ['const_threshold_100', 'const_threshold_80']
    filesnames = ['growing_erank_100', 'growing_erank_80']
    for i in range(2):
        fig, ax = plt.subplots(figsize=(width, height), dpi=100)

        multi_rank_plot(ax, names[i], sample_no=None)

        ax.set_ylim([0.45, 1.0])
        ax.set_ylabel('relative erank $r$')
        ax.set_xlabel('steps')
        ax.legend(loc='lower right')

        savefig(fig, f'{filesnames[i]}')
        plt.show()

    name = 'const_threshold_80'
    filename = 'growing_channels_80'
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)

    multi_simple_plot(ax, name, 'out_channel', sample_no=None)

    ax.set_ylim([0, 22.5])
    ax.set_yticks(np.arange(0, 22, 5))
    ax.set_ylabel('number of channels $n$')
    ax.set_xlabel('steps')
    ax.legend(loc='lower right')

    savefig(fig, filename)
    plt.show()

def acc_over_params():
    width, height = 4, 4
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    ax.set_xscale('log')
    ax.grid(axis='y', color='0.90')

    name = f'baseline'
    logs = load_logs(name)
    x, y = logs['num_params'], logs['val_acc']
    x, y = np.array(x), np.array(y)
    ax.scatter(x, y, marker='s', label=f'baseline')
    # ax.annotate(f'baseline', xy=(np.mean(x), np.mean(y)), xytext=(-5, 10), textcoords='offset points', ha='right')

    name = f'reproduce_gamma_90'
    logs = load_logs(name)
    x, y = logs['num_params'], logs['val_acc']
    x, y = np.array(x), np.array(y)
    ax.scatter(x, y, marker='v', label=f'NORTH-Select')
    # ax.annotate(f'NORTH-Select', xy=(np.mean(x), np.mean(y)), xytext=(-5, 10), textcoords='offset points', ha='right')

    thresholds = [30, 40, 50, 60, 70, 80, 90, 100]
    for i, t in enumerate(thresholds):
        name = f'const_threshold_{t}'
        color = i+2
        logs = load_logs(name)
        x, y = logs['num_params'], logs['val_acc']
        x, y = np.array(x), np.array(y)
        label = f'q=.{t//10}' if t != 100 else 'q=1'
        ax.scatter(x, y, label=label)
        if not t == 100:
            confidence_ellipse(x, y, ax, n_std=1.0, edgecolor=f'C{color}', alpha=0.2, facecolor=f'C{color}')
        ax.annotate(label, xy=(np.mean(x), np.mean(y)), xytext=(-10, -20), textcoords='offset points')

    ax.set_ylabel('accuracy')
    ax.set_xlabel('number of parameters')

    legend_baseline = mlines.Line2D([], [], color='C0', marker='s', linestyle='None', label='baseline')
    legend_north_select = mlines.Line2D([], [], color='C1', marker='v', linestyle='None', label='NORTH-Select${}^2$')
    legend_q = mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='my variation')
    ax.legend(handles=[legend_baseline, legend_north_select, legend_q])

    savefig(fig, 'acc_over_params')
    plt.show()


def print_logs(name, label=None):
    logs = load_logs(name)
    cfg = load_cfg(name)
    if label is None:
        s = name
    else:
        s = label

    mean_channels = []
    for i in range(3):
        channels = [run[-1] for run in logs[f'conv{i}_out_channel']]
        mean_channels.append(np.mean(channels))
    mean_channels = [round(c) for c in mean_channels]
    s += ' & ' + '-'.join([str(c) for c in mean_channels])

    mean_params = np.mean(logs['num_params'])
    std_params = np.std(logs['num_params'])
    s += f' & {mean_params/1000:3.1f}({std_params/1000:3.1f})'

    mean_acc = np.mean(logs['val_acc'])
    std_acc = np.std(logs['val_acc'])
    s += f' & {mean_acc*100:2.1f}({std_acc*100:2.1f})\\%'
    s += r' \\'

    print(s)


def print_sweep_table():
    print_logs('baseline')
    print_logs('reproduce_gamma_90', 'NORTH-Select')
    thresholds = [30, 40, 50, 60, 70, 80, 90, 100]
    for t in reversed(thresholds):
        name = f'const_threshold_{t}'
        print_logs(name, f'q={t/100:1.1f}')

def print_final_table():
    print_logs('test_baseline')
    print_logs('test_baseline_opt_cfg')
    print_logs('test_opt_cfg')
    print_logs('test_opt_cfg_static')


def final_acc():
    width, height = 4, 4
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    ax.grid(axis='y', color='0.90')

    names = ['test_baseline', 'test_baseline_opt_cfg', 'test_opt_cfg', 'test_opt_cfg_static']
    labels = ['baseline', 'BOHB on\nstatic ConvNet', 'BOHB on\ngrowing ConvNet', 'grown ConvNets\nretrained']
    data = []
    for i, name in enumerate(names):
        logs = load_logs(name)
        acc = logs['val_acc']
        data.append(acc)
        ax.scatter([i+1]*len(acc), acc, label=labels[i])

    ax.boxplot(data)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel('accuracy')
    # ax.legend()

    # names[0] is baseline
    alpha = 0.05
    null_name = 'baseline'
    null_mean = np.mean(data[0])
    for i in range(1, len(names)):
        ztest_Score, p_value = ztest(data[i], value=null_mean, alternative='larger')
        if p_value < alpha:
            result = 'rejected'
        else:
            result = 'NOT rejected'
        print(f'{names[i]} same distribution as {null_name} (with alpha={alpha}): {result}')

    savefig(fig, 'final_acc')
    plt.show()


if __name__ == '__main__':
    erank_vs_edim()
    baseline_erank()
    growing()
    acc_over_params()
    print_sweep_table()
    print_final_table()
    final_acc()
