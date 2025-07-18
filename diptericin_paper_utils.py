import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



"""functions"""
def style_axes(ax, fontsize=24):
    plt.minorticks_off()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    plt.tight_layout()
    
    return ax


def emd_from_line_dist(y1, y2, bins=None):
    """from two unnormalized 1D line distributions (e.g., transcript counts along ap), 
    compute the Earth Mover's Distance (Wasserstein-1 distance) by first normalizing by
    total mass, computing the cumulative distribution functions, and summing the 
    absolute differences.
    
    y1, y2: 1D arrays of line distribution (same length)
    bins: bin centers on a regular grid. len(bins) = len(y1) + 1"""
    
    assert len(y1) == len(y2)
    if bins is None:
        bins == np.arange(len(y1))
    else:
        assert len(y1) == len(bins)
    
    bin_width = np.diff(bins)[0]
    # normalize by total mass
    y1 = y1 / np.sum(y1) / bin_width
    y2 = y2 / np.sum(y2) / bin_width
    
    # compute CDFs
    Y1 = np.cumsum(y1 * bin_width)
    Y2 = np.cumsum(y2 * bin_width)
    
    # compute EMD as sum of absolute differences
    emd = np.sum(np.abs(Y1 - Y2) * bin_width)
    
    return emd
    

def bootstrap_wass1_from_means(ld1, ld2, bins=None, n_bootstraps=10):
    wass1s = np.zeros(n_bootstraps)
    for n in range(n_bootstraps):
        ids1 = np.random.choice(len(ld1), len(ld1))
        ids2 = np.random.choice(len(ld2), len(ld2))
        y1 = np.mean(ld1[ids1], axis=0)
        y2 = np.mean(ld2[ids2], axis=0)
        
        wass1s[n] = emd_from_line_dist(y1, y2, bins=bins)
            
    return wass1s


"""params"""
dpt = 'DptA'

colors = {'no_inj': [0.8, 0.8, 0.8],
          'mock': [0.4, 0.4, 0.4],
          'e.coli': [0, 0.4, 0],
          'complete': [0, 0.8, 0],
          'bacteria': [0.8, 0, 0.8],
          'dye': np.array([204., 85., 0.]) / 255
}

linewidth = 4
fontsize = 24
rc_params = {'font.family': 'Arial',
          'axes.linewidth': linewidth,
          'font.size': fontsize}

standard_ap = np.linspace(0, 1, 15)
standard_ap = np.concatenate((standard_ap, np.array([standard_ap[-1] + np.diff(standard_ap)[0]])))
fb_start_fraction = 0.1#0.18
fb_start_id = int(np.where(standard_ap > fb_start_fraction)[0][0]) - 1

