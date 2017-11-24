import numpy as np
import itertools
from astropy.io import fits
from astropy import units as u
import sys
from .progressbar import AnimatedProgressBar
from .verbose_output import print_to_terminal
from io import *

def get_moments(cube, ppv_vol, rms_approx, sigma_cut, write_moments, dir, filename, verbose):
    """
    Create moment maps
    """

    if verbose:
        progress_bar = print_to_terminal(stage='s1', step='moments')

    # If upper and lower limits are imposed on the velocity range
    if (ppv_vol[0] != 0) & (ppv_vol[1] != 0):
        momzero = cube.with_mask(cube > u.Quantity(
            rms_approx * sigma_cut, cube.unit)).spectral_slab(ppv_vol[0]*u.km/u.s,ppv_vol[1]*u.km/u.s).moment0(axis=0)
        momone = cube.with_mask(cube > u.Quantity(
            rms_approx * sigma_cut, cube.unit)).spectral_slab(ppv_vol[0]*u.km/u.s,ppv_vol[1]*u.km/u.s).moment1(axis=0)
        momtwo = cube.with_mask(cube > u.Quantity(
            rms_approx * sigma_cut, cube.unit)).spectral_slab(ppv_vol[0]*u.km/u.s,ppv_vol[1]*u.km/u.s).linewidth_sigma()

    # If no velocity limits are imposed
    else:
        momzero = cube.with_mask(cube > u.Quantity(
            rms_approx * sigma_cut, cube.unit)).moment0(axis=0)
        momone = cube.with_mask(cube > u.Quantity(
            rms_approx * sigma_cut, cube.unit)).moment1(axis=0)
        momtwo = cube.with_mask(cube > u.Quantity(
            rms_approx * sigma_cut, cube.unit)).linewidth_sigma()

    # Write moments
    if write_moments:
        output_moments(momzero, momone, momtwo, dir, filename)

    return momzero, momone, momtwo

def get_coverage(momzero, spacing):
    """
    Returns locations of SAAss
    """
    cols, rows = np.where(momzero != 0.0)

    rangex = [np.min(rows), np.max(rows)]
    sizex = np.abs(np.min(rows)-np.max(rows))
    rangey = [np.min(cols), np.max(cols)]
    sizey = np.abs(np.min(cols)-np.max(cols))

    nposx = int((sizex/(spacing*2.))+1.0)
    nposy = int((sizey/(spacing*2.))+1.0)

    cov_x = np.max(rangex)-(spacing*2.)*np.arange(nposx)
    cov_y = np.min(rangey)+(spacing*2.)*np.arange(nposy)

    return cov_x, cov_y

def define_coverage(cube, momzero, rsaa, verbose):
    """
    Returns locations of SAAs which contain significant information and computes
    a spatially-averaged spectrum.
    """

    spacing = rsaa/2.
    cov_x, cov_y = get_coverage(momzero, spacing)

    coverage = np.full([len(cov_y)*len(cov_x),2], np.nan)
    spec = np.full([cube.shape[0], len(cov_y), len(cov_x)], np.nan)

    count= 0.0
    if verbose:
        progress_bar = print_to_terminal(stage='s1', step='coverage', length=len(cov_y)*len(cov_x))

    for cx,cy in itertools.product(cov_x, cov_y):
        if verbose and (count % 1 == 0):
                progress_bar + 1
                progress_bar.show_progress()

        idx = int((cov_x[0]-cx)/rsaa)
        idy = int((cy-cov_y[0])/rsaa)

        limx = [int(cx-spacing*2.), int(cx+spacing*2.)]
        limy = [int(cy-spacing*2.), int(cy+spacing*2.)]
        limx = [lim if (lim > 0) else 0 for lim in limx ]
        limx = [lim if (lim < np.shape(momzero)[1]) else np.shape(momzero)[1] for lim in limx ]
        limy = [lim if (lim > 0) else 0 for lim in limy ]
        limy = [lim if (lim < np.shape(momzero)[0]) else np.shape(momzero)[0] for lim in limy ]

        momzero_cutout = momzero[min(limy):max(limy),
                                 min(limx):max(limx)]

        finite = np.isfinite(momzero_cutout)
        nmask = np.count_nonzero(finite)
        if nmask > 0:
            tot_non_zero = np.count_nonzero(np.isfinite(momzero_cutout) & (momzero_cutout!=0))
            fraction = tot_non_zero / nmask
            if fraction > 0.5:
                coverage[idy+(idx*len(cov_y)),:] = cx,cy
                spec[:, idy, idx] = cube[:,
                                         min(limy):max(limy),
                                         min(limx):max(limx)].mean(axis=(1,2))

    if verbose:
        print('')

    return coverage, spec

def get_random_saa(cc, ss, samplesize, rsaa, verbose=False):
    """
    Get a randomly selected sample of spectral averaging areas
    """

    if verbose:
        print('')
        print("Extracting randomly sampled SAAs for training set...")

    npixpersaa = (rsaa[0]*2.0)**2.0
    training_set_size = npixpersaa*samplesize

    _ss = np.asarray(ss)
    _cc = np.asarray(cc)

    sample_ss = np.full(np.shape(_ss), np.nan)
    sample_cc = np.full(np.shape(_cc), np.nan)

    cc_finite = (np.isfinite(_cc[:,0]))
    idx = np.squeeze(np.where(cc_finite == True))

    low=0
    high=np.size(idx)
    randpix = np.random.randint(low, high, size=samplesize)
    idx=idx[randpix]
    idx_ss = np.array(np.unravel_index(idx, (np.shape(_ss)[2], np.shape(_ss)[1])))

    for i in range(0, len(idx)):
        sample_ss[:,idx_ss[1,i], idx_ss[0,i]] = _ss[:,idx_ss[1,i], idx_ss[0,i]]
        sample_cc[idx[i], :]=_cc[idx[i], :]

    if verbose:
        print('Training set size = {}'.format(int(training_set_size)))
        if training_set_size < 1000.0:
            print('WARNING: Training set size {} < 1000, try increasing the sample size (for equivalent RSAA)'.format(int(training_set_size)))
        print('')

    sample_ss = [sample_ss]

    return sample_cc, sample_ss

def plot_rsaa(coverage_coordinates, momzero, rsaa, dir, filename):
    """
    Plot the SAA boxes
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig = plt.figure(1, figsize=(10.0, 4.0))
    fig.clf()
    ax = fig.add_subplot(111)
    plt.imshow(momzero, cmap='Greys', origin='lower',
               interpolation='nearest')
    cols = ['black', 'red', 'blue']
    size = [0.5, 1, 2]
    alpha = [1, 0.8, 0.5]
    for i, (r, covcoords) in enumerate(zip(rsaa, coverage_coordinates)):
        for j in range(covcoords.shape[0]):
            if np.all(np.isfinite(covcoords[j, :])):
                ax.add_patch(
                    patches.Rectangle(
                        (covcoords[j, 0] - r, covcoords[j, 1] - r),
                        r * 2., r * 2., facecolor='none',
                        edgecolor=cols[i], lw=size[i], alpha=alpha[i]))

    plt.savefig(dir+'/'+filename+'coverage.pdf', dpi=600,bbox_inches='tight')
    plt.draw()
    plt.show()
