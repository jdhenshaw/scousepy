# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np
import itertools
from astropy.io import fits
from astropy import units as u
import sys
from .progressbar import AnimatedProgressBar
from .verbose_output import print_to_terminal
from .io import *
import random

def get_moments(self, write_moments, dir, filename, verbose):
    """
    Create moment maps
    """

    if verbose:
        progress_bar = print_to_terminal(stage='s1', step='moments')

    # If upper and lower limits are imposed on the velocity range
    if (self.ppv_vol[0] != 0) & (self.ppv_vol[1] != 0):
        momzero = self.cube.with_mask(self.cube > u.Quantity(
            self.rms_approx * self.sigma_cut, self.cube.unit)).spectral_slab(self.ppv_vol[0]*u.km/u.s,self.ppv_vol[1]*u.km/u.s).moment0(axis=0)
        momone = self.cube.with_mask(self.cube > u.Quantity(
            self.rms_approx * self.sigma_cut, self.cube.unit)).spectral_slab(self.ppv_vol[0]*u.km/u.s,self.ppv_vol[1]*u.km/u.s).moment1(axis=0)
        momtwo = self.cube.with_mask(self.cube > u.Quantity(
            self.rms_approx * self.sigma_cut, self.cube.unit)).spectral_slab(self.ppv_vol[0]*u.km/u.s,self.ppv_vol[1]*u.km/u.s).linewidth_sigma()
        slab = self.cube.with_mask(self.cube > u.Quantity(
            self.rms_approx * self.sigma_cut, self.cube.unit)).spectral_slab(self.ppv_vol[0]*u.km/u.s,self.ppv_vol[1]*u.km/u.s)
        momnine = np.empty(np.shape(momone))
        momnine.fill(np.nan)
        idxmax= np.argmax(slab._data, axis=0)
        momnine=slab.spectral_axis[idxmax].value
        idnan = (np.isfinite(momtwo.value)==0)
        momnine[idnan] = np.nan
        momnine = momnine * u.km/u.s

    # If no velocity limits are imposed
    else:
        momzero = self.cube.with_mask(self.cube > u.Quantity(
            self.rms_approx * self.sigma_cut, self.cube.unit)).moment0(axis=0)
        momone = self.cube.with_mask(self.cube > u.Quantity(
            self.rms_approx * self.sigma_cut, self.cube.unit)).moment1(axis=0)
        momtwo = self.cube.with_mask(self.cube > u.Quantity(
            self.rms_approx * self.sigma_cut, self.cube.unit)).linewidth_sigma()
        slab = self.cube.with_mask(self.cube > u.Quantity(
            self.rms_approx * self.sigma_cut, self.cube.unit))
        momnine = np.empty(np.shape(momone))
        momnine.fill(np.nan)
        idxmax= np.argmax(slab._data, axis=0)
        momnine=slab.spectral_axis[idxmax].value
        idnan = (np.isfinite(momtwo.value)==0)
        momnine[idnan] = np.nan
        momnine = momnine * u.km/u.s

    # Write moments
    if write_moments:
        output_moments(momzero, momone, momtwo, momnine, dir, filename)

    return momzero, momone, momtwo, momnine

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

def define_coverage(cube, momzero, momzero_mod, rsaa, nrefine, verbose, redefine=False):
    """
    Returns locations of SAAs which contain significant information and computes
    a spatially-averaged spectrum.
    """

    spacing = rsaa/2.
    cov_x, cov_y = get_coverage(momzero, spacing)

    coverage = np.full([len(cov_y)*len(cov_x),2], np.nan)
    spec = np.full([cube.shape[0], len(cov_y), len(cov_x)], np.nan)
    ids = np.full([len(cov_y)*len(cov_x),cube.shape[2]*cube.shape[1], 2], np.nan)
    frac = np.full([len(cov_y)*len(cov_x)], np.nan)

    count= 0.0
    if not redefine:
        if verbose:
            progress_bar = print_to_terminal(stage='s1', step='coverage', length=len(cov_y)*len(cov_x))

    for cx,cy in itertools.product(cov_x, cov_y):
        if not redefine:
            if verbose and (count % 1 == 0):
                progress_bar + 1
                progress_bar.show_progress()

        idx = int((cov_x[0]-cx)/rsaa)
        idy = int((cy-cov_y[0])/rsaa)

        limx = [int(cx-spacing*2.), int(cx+spacing*2.)]
        limy = [int(cy-spacing*2.), int(cy+spacing*2.)]
        limx = [lim if (lim > 0) else 0 for lim in limx ]
        limx = [lim if (lim < np.shape(momzero)[1]-1) else np.shape(momzero)[1]-1 for lim in limx ]
        limy = [lim if (lim > 0) else 0 for lim in limy ]
        limy = [lim if (lim < np.shape(momzero)[0]-1) else np.shape(momzero)[0]-1 for lim in limy ]

        rangex = range(min(limx), max(limx)+1)
        rangey = range(min(limy), max(limy)+1)

        momzero_cutout = momzero_mod[min(limy):max(limy),
                                     min(limx):max(limx)]

        finite = np.isfinite(momzero_cutout)
        nmask = np.count_nonzero(finite)
        if nmask > 0:
            tot_non_zero = np.count_nonzero(np.isfinite(momzero_cutout) & (momzero_cutout!=0))
            fraction = tot_non_zero / nmask
            if redefine:
                lim = 0.5/nrefine
            else:
                lim = 0.5
            if fraction >= lim:
                frac[idy+(idx*len(cov_y))] = fraction
                coverage[idy+(idx*len(cov_y)),:] = cx,cy
                spec[:, idy, idx] = cube[:,
                                         min(limy):max(limy),
                                         min(limx):max(limx)].mean(axis=(1,2))
                count=0
                for i in rangex:
                    for j in rangey:
                        ids[idy+(idx*len(cov_y)), count, 0],\
                        ids[idy+(idx*len(cov_y)), count, 1] = j, i
                        count+=1
    if verbose:
        print('')

    return coverage, spec, ids, frac

def get_rsaa(self):
    rsaa = []
    for i in range(1, int(self.nrefine)+1):
        newrsaa = self.rsaa[0]/i
        if newrsaa > 0.5:
            rsaa.append(newrsaa)
    return rsaa

def get_random_saa(cc, samplesize, r, verbose=False):
    """
    Get a randomly selected sample of spectral averaging areas
    """

    if verbose:
        print('')
        print("Extracting randomly sampled SAAs for training set...")

    npixpersaa = (r*2.0)**2.0
    training_set_size = npixpersaa*samplesize

    sample = np.sort(random.sample(range(0,len(cc[:,0])), samplesize))

    if verbose:
        print('Training set size = {}'.format(int(training_set_size)))
        if training_set_size < 1000.0:
            print('WARNING: Training set size {} < 1000, try increasing the sample size (for equivalent RSAA)'.format(int(training_set_size)))
        print('')

    return sample

def plot_rsaa(dict, momzero, rsaa, dir, filename):
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
    cols = ['blue', 'red', 'yellow', 'limegreen', 'cyan', 'magenta']

    for i, r in enumerate(rsaa, start=0):
        for j in range(len(dict[i].keys())):
            if dict[i][j].to_be_fit:
                ax.add_patch(patches.Rectangle(
                            (dict[i][j].coordinates[0] - r, \
                             dict[i][j].coordinates[1] - r),\
                             r * 2., r * 2., facecolor=cols[i],
                             edgecolor=cols[i], lw=0.1, alpha=0.1))
                ax.add_patch(patches.Rectangle(
                            (dict[i][j].coordinates[0] - r, \
                             dict[i][j].coordinates[1] - r),\
                             r * 2., r * 2., facecolor='None',
                             edgecolor=cols[i], lw=0.2, alpha=0.25))

    plt.savefig(dir+'/'+filename+'_coverage.pdf', dpi=600,bbox_inches='tight')
    plt.draw()
    plt.show()

def calculate_delta_v(self, momone, momnine):

    # Generate an empty array
    delta_v = np.empty(np.shape(momone))
    delta_v.fill(np.nan)
    delta_v = np.abs(momone.value-momnine.value)

    return delta_v

def generate_steps(self, delta_v):
    """
    Creates logarithmically spaced lag values for structure function computation
    """
    median = np.nanmedian(delta_v)
    step_values = np.logspace(np.log10(median), \
                              np.log10(np.nanmax(delta_v)), \
                              self.nrefine )
    return list(step_values)

def refine_momzero(self, momzero, delta_v, minval, maxval):
    """
    Refines momzero based on upper/lower lims of delta_v
    """
    mom_zero=None
    keep = ((delta_v >= minval) & (delta_v <= maxval))
    mom_zero = np.zeros(np.shape(momzero))
    mom_zero[keep] = momzero[keep]

    return mom_zero
