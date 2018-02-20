# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import pyspeckit
import warnings
from astropy import log
from matplotlib import pyplot

from .interactiveplot import showplot
from .stage_3 import argsort, get_spec, get_flux

def interactive_plot(self, blocksize=7, figsize = None, plot_residuals=False,
                     blockrange=None):
    """
    Generate an interactive plot so the user can select fits they would like to
    take a look at again.
    """

    check_spec_indices = []

    # Generate blocks and masks
    nxblocks, nyblocks, blockarr = get_blocks(self, blocksize)
    nblocks = nxblocks*nyblocks
    fit_mask = pad_fits(self, blocksize, nxblocks, nyblocks)
    spec_mask = pad_spec(self, blocksize, nxblocks, nyblocks)

    # For staged checking
    if blockrange==None:
        blockrange=np.arange(0,int(nblocks))
    else:
        if np.max(blockrange)>int(nblocks):
            blockrange=np.arange(int(np.min(blockrange)),int(nblocks))
        else:
            blockrange=np.arange(int(np.min(blockrange)),int(np.max(blockrange)))

    # Cycle through the blocks
    for i in blockrange:
        blocknum = i+1
        keep = (blockarr == blocknum)
        speckeys = spec_mask[keep]
        fitkeys = fit_mask[keep]

        # We are only interested in blocks where there is at least 1 model
        # solution - don't bother with the others
        if np.any(np.isfinite(fitkeys)):

            if figsize is None:
                figsize = [14,10]
            else:
                figsize=figsize
            # Prepare plot
            fig, ax = pyplot.subplots(blocksize, blocksize, figsize=figsize)
            ax = np.flip(ax,0)
            ax = [a for axis in ax for a in axis]

            # Cycle through the spectra contained within the block
            for j in range(np.size(speckeys)):

                # Firstly check to see if the spectrum is located within the
                # fit dictionary
                key = speckeys[j]
                keycheck = key in self.indiv_dict.keys()

                # If so - plot the spectrum and its model solution
                if keycheck:
                    spectrum = self.indiv_dict[key]
                    # Get the correct subplot axis
                    axis = ax[j]
                    # First plot the Spectrum
                    axis.plot(self.xtrim, get_flux(self, spectrum), 'k-', drawstyle='steps', lw=1)
                    # Recreate the model from information held in the solution
                    # description
                    bfmodel = spectrum.model
                    mod, res = recreate_model(self, spectrum, bfmodel)
                    # now overplot the model
                    if bfmodel.ncomps == 0.0:
                        axis.plot(self.xtrim, mod[:,0], 'b-', lw=1)
                    else:
                        for k in range(int(bfmodel.ncomps)):
                            axis.plot(self.xtrim, mod[:,k], 'b-', lw=1)
                    if plot_residuals:
                        axis.plot(self.xtrim, res,'g-', drawstyle='steps', lw=1)

            # Create the interactive plot
            intplot = showplot(fig, ax)

            # Get the indices of the spectra we want to take another look at
            check_spec = get_indices(intplot, speckeys)

            # append
            check_spec_indices.append(check_spec)

    # Now flatten
    check_spec_indices = [idx for indices in check_spec_indices for idx in indices]
    if np.size(check_spec_indices) > 0.0:
        check_spec_indices = np.array(check_spec_indices)
        sortidx = argsort(check_spec_indices)
        check_spec_indices = check_spec_indices[sortidx]
    else:
        check_spec_indices = np.array([])

    return check_spec_indices

def get_indices(plot, speckeys):
    """
    Returns indices of spectra we want to take a closer look at
    """

    subplots = plot.subplots
    check_spec = speckeys[subplots]

    return check_spec

def get_blocks(self, blocksize):
    """
    Break the map up into blocks for plotting the spectra as they appear on the
    sky
    """

    # Get the correct number of blocks - this can be controlled by the user with
    # the keyword blocksize
    blocksize=int(blocksize)
    nyblocks = np.shape(self.cube)[1]/blocksize
    if (nyblocks-int(nyblocks))==0.0:
        nyblocks = round(nyblocks)
    else:
        nyblocks = round(nyblocks+0.5)
    nxblocks = np.shape(self.cube)[2]/blocksize
    if (nxblocks-int(nxblocks))==0.0:
        nxblocks = round(nxblocks)
    else:
        nxblocks = round(nxblocks+0.5)

    # Arrange the map into blocks of particular size - Here we essentially pad
    # the map to get filled blocks - this helps with the plotting
    blockarr = np.zeros([nyblocks*blocksize,nxblocks*blocksize], dtype='int')
    for i in range(np.shape(blockarr)[1]):
        for j in range(np.shape(blockarr)[0]):
            xbin = int(((i)/blocksize))
            ybin = int(((j)/blocksize))
            blockval = ybin + nyblocks*xbin + 1
            blockarr[j,i] = blockval

    return nxblocks, nyblocks, blockarr

def pad_spec(self, blocksize, nxblocks, nyblocks):
    """
    Returns a mask containing keys indicating where we have best-fitting
    solutions

    Notes:
    This is padded to fill blocks for the interactive plotting - same as pad
    fits - its a little clunky and could no doubt be improved, but it does the
    job

    """
    spec_mask = np.full([nyblocks*blocksize,nxblocks*blocksize], np.nan)
    for i in range(np.shape(self.cube)[2]):
        for j in range(np.shape(self.cube)[1]):
            key = j+i*(np.shape(self.cube)[1])
            if key in self.indiv_dict:
                if self.indiv_dict[key].model.ncomps > 0.0:
                    spec_mask[j,i]=key
                else:
                    spec_mask[j,i]=key
    return spec_mask

def pad_fits(self, blocksize, nxblocks, nyblocks):
    """
    Returns a mask containing keys indicating where we have best-fitting
    solutions which have ncomps != 0.0 - i.e. where we actually have fits

    Notes:
    This is padded to fill blocks for the interactive plotting - same as pad
    fits - its a little clunky and could no doubt be improved, but it does the
    job

    """

    fit_mask = np.full([nyblocks*blocksize,nxblocks*blocksize], np.nan)
    for i in range(np.shape(self.cube)[2]):
        for j in range(np.shape(self.cube)[1]):
            key = j+i*(np.shape(self.cube)[1])
            if key in self.indiv_dict:
                if self.indiv_dict[key].model.ncomps > 0.0:
                    fit_mask[j,i]=key
    return fit_mask

def recreate_model(self, spectrum, bf, alternative=False):
    """
    Recreates model from parameters
    """

    # Make pyspeckit be quiet
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        old_log = log.level
        log.setLevel('ERROR')
        # generate a spectrum
        spec = get_spec(self, spectrum)
        spec.specfit.fittype = bf.fittype
        spec.specfit.fitter = spec.specfit.Registry.multifitters[bf.fittype]
        if bf.ncomps != 0.0:
            mod = np.zeros([len(self.xtrim), int(bf.ncomps)])
            for k in range(int(bf.ncomps)):
                modparams = bf.params[(k*len(bf.parnames)):(k*len(bf.parnames))+len(bf.parnames)]
                mod[:,k] = spec.specfit.get_model_frompars(self.xtrim, modparams)
            totmod = np.sum(mod, axis=1)
            res = get_flux(self, spectrum)-totmod
        else:
            mod = np.zeros([len(self.xtrim), 1])
            res = get_flux(self, spectrum)
        log.setLevel(old_log)

    return mod, res

def check_and_flatten(self, check_spec_indices):
    """
    Checks to see if staged_fitting has been initiated. If so it appends
    current list to the existing one.
    """

    _check_spec_indices=None

    if np.size(self.check_spec_indices)!=0:
        _check_spec_indices = [list(self.check_spec_indices) + list(check_spec_indices)]
        _check_spec_indices = np.asarray(_check_spec_indices)
        _check_spec_indices = np.unique(_check_spec_indices) # Just in case
    else:
        _check_spec_indices = check_spec_indices

    return _check_spec_indices
