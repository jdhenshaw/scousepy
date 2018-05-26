# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pyspeckit
import warnings
import time
import string

from astropy import log
from astropy.io import fits
from astropy.utils.console import ProgressBar
from matplotlib import pyplot

from .interactiveplot import showplot, InteractivePlot
from .stage_3 import argsort, get_flux

def interactive_plot(scouseobject, blocksize=7, figsize=None, plot_residuals=False,
                     blockrange=None):
    """
    Generate an interactive plot so the user can select fits they would like to
    take a look at again.
    """

    check_spec_indices = []
    check_block_indices = []

    # Generate blocks and masks
    nxblocks, nyblocks, blockarr = get_blocks(scouseobject, blocksize)
    nblocks = nxblocks*nyblocks
    fit_mask = pad_fits(scouseobject, blocksize, nxblocks, nyblocks)
    spec_mask = pad_spec(scouseobject, blocksize, nxblocks, nyblocks)

    # For staged checking
    if blockrange is None:
        blockrange=np.arange(0,int(nblocks))
    else:
        if np.max(blockrange)>int(nblocks):
            blockrange=np.arange(int(np.min(blockrange)),int(nblocks)+1)
        else:
            blockrange=np.arange(int(np.min(blockrange)),int(np.max(blockrange)))

    # Cycle through the blocks
    #for i in blockrange:
    #    blocknum = i+1

    def callback_check_spec(blocknum, intplot):
        keep = (blockarr == blocknum)
        speckeys = spec_mask[keep]

        # Get the indices of the spectra we want to take another look at
        check_spec = get_indices(intplot, speckeys)

        # append
        if np.size(check_spec) == blocksize**2:
            check_block_indices.append(blocknum)
        else:
            check_spec_indices.append(check_spec)

        for ax in axes_flat:
            ax.cla()
            [xx.set_visible(False) for xx in ax.get_yticklabels()]
            [xx.set_visible(False) for xx in ax.get_xticklabels()]

    def plot_blocknum(blocknum, intplot):

        keep = (blockarr == blocknum)
        speckeys = spec_mask[keep]
        fitkeys = fit_mask[keep]

        # We are only interested in blocks where there is at least 1 model
        # solution - don't bother with the others
        if np.any(np.isfinite(fitkeys)):
            print("Checking block {0}".format(blocknum))

            # Cycle through the spectra contained within the block
            for j in range(np.size(speckeys)):

                # Firstly check to see if the spectrum is located within the
                # fit dictionary
                key = speckeys[j]
                keycheck = key in scouseobject.indiv_dict.keys()

                # If so - plot the spectrum and its model solution
                if keycheck:
                    spectrum = scouseobject.indiv_dict[key]
                    # Get the correct subplot axis
                    axis = axes_flat[j]

                    # clear the axes
                    # (this is now handled in the other callback)
                    #axis.cla()

                    # First plot the Spectrum
                    axis.plot(scouseobject.xtrim, get_flux(scouseobject, spectrum), 'k-',
                              drawstyle='steps', lw=1)
                    # Recreate the model from information held in the solution
                    # description
                    bfmodel = spectrum.model
                    mod, res = recreate_model(scouseobject, spectrum, bfmodel)
                    totmod = mod.sum(axis=1)
                    # now overplot the model
                    if bfmodel.ncomps == 0.0:
                        axis.plot(scouseobject.xtrim, mod[:,0], 'b-', lw=1)
                    else:
                        for k in range(int(bfmodel.ncomps)):
                            axis.plot(scouseobject.xtrim, mod[:,k], 'b-', lw=1)
                    axis.plot(scouseobject.xtrim, totmod, 'r-', lw=1, zorder=-5)
                    if plot_residuals:
                        axis.plot(scouseobject.xtrim, res,'g-', drawstyle='steps', lw=1)
                    axis.get_xaxis().set_ticks([])
                    axis.get_yaxis().set_ticks([])

            return True
        else:
            print("Nothing to plot for this block; no fits available.")
            return True

    # set up the plot window *once*
    if figsize is None:
        figsize = [14,10]
    # Prepare plot
    fig, axes = pyplot.subplots(blocksize, blocksize, figsize=figsize)
    axes_flat = [a for axis in axes[::-1] for a in axis]

    for ax in axes_flat:
        [xx.set_visible(False) for xx in ax.get_yticklabels()]
        [xx.set_visible(False) for xx in ax.get_xticklabels()]

    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    intplot = showplot(fig, axes_flat, blocknum_ind=0,
                       blockrange=blockrange,
                       callback=plot_blocknum,
                       callback_check_spec=callback_check_spec,
                      )

    # wait until all the plots have been looped over
    while not intplot.done:
        try:
            # using just a few little bits of plt.pause below
            plt.gcf().canvas.draw()
            plt.gcf().canvas.start_event_loop(0.1)
            time.sleep(0.1)
        except KeyboardInterrupt:
            break
    print("")

    plt.close(intplot.fig.number)
    plt.close(fig.number)

    # Now flatten
    check_spec_indices = [idx for indices in check_spec_indices for idx in indices]
    if np.size(check_spec_indices) > 0.0:
        check_spec_indices = np.array(check_spec_indices)
        sortidx = argsort(check_spec_indices)
        check_spec_indices = check_spec_indices[sortidx]
        check_spec_indices = list(check_spec_indices)
    else:
        #check_spec_indices = np.array([])
        check_spec_indices = []

    if np.size(check_block_indices) > 0.0:
        #check_block_indices = np.array(check_block_indices)
        check_block_indices = list(check_block_indices)
    else:
        #check_block_indices = np.array([])
        check_block_indices = []

    return check_spec_indices, check_block_indices

def get_indices(plot, speckeys):
    """
    Returns indices of spectra we want to take a closer look at
    """

    subplots = plot.subplots
    check_spec = speckeys[subplots]

    return check_spec

def get_blocks(scouseobject, blocksize):
    """
    Break the map up into blocks for plotting the spectra as they appear on the
    sky
    """

    # Get the correct number of blocks - this can be controlled by the user with
    # the keyword blocksize
    blocksize=int(blocksize)
    nyblocks = np.shape(scouseobject.cube)[1]/blocksize
    if (nyblocks-int(nyblocks))==0.0:
        nyblocks = round(nyblocks)
    else:
        nyblocks = round(nyblocks+0.5)
    nxblocks = np.shape(scouseobject.cube)[2]/blocksize
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

def pad_spec(scouseobject, blocksize, nxblocks, nyblocks):
    """
    Returns a mask containing keys indicating where we have best-fitting
    solutions

    Notes:
    This is padded to fill blocks for the interactive plotting - same as pad
    fits - its a little clunky and could no doubt be improved, but it does the
    job

    """
    spec_mask = np.full([nyblocks*blocksize,nxblocks*blocksize], np.nan)

    shape = scouseobject.cube.shape[1:]

    yinds, xinds = np.indices(shape)
    yinds_f, xinds_f = yinds.ravel(), xinds.ravel()
    flat_inds = np.ravel_multi_index(np.array([yinds_f, xinds_f]), shape)
    ok_inds = np.array([key in scouseobject.indiv_dict for key in flat_inds], dtype='bool').reshape(shape)
    spec_mask[:shape[0], :shape[1]][ok_inds] = flat_inds.reshape(shape)[ok_inds]

    return spec_mask

def pad_fits(scouseobject, blocksize, nxblocks, nyblocks):
    """
    Returns a mask containing keys indicating where we have best-fitting
    solutions which have ncomps != 0.0 - i.e. where we actually have fits

    Notes:
    This is padded to fill blocks for the interactive plotting - same as pad
    fits - its a little clunky and could no doubt be improved, but it does the
    job

    """
    fit_mask = np.full([nyblocks*blocksize,nxblocks*blocksize], np.nan)

    shape = scouseobject.cube.shape[1:]

    yinds, xinds = np.indices(shape)
    yinds_f, xinds_f = yinds.ravel(), xinds.ravel()
    flat_inds = np.ravel_multi_index(np.array([yinds_f, xinds_f]), shape)
    ok_inds = np.array([(key in scouseobject.indiv_dict) and (scouseobject.indiv_dict[key].model.ncomps > 0)
                        for key in flat_inds], dtype='bool').reshape(shape)
    fit_mask[:shape[0], :shape[1]][ok_inds] = flat_inds.reshape(shape)[ok_inds]

    return fit_mask

def recreate_model(scouseobject, spectrum, bf, alternative=False):
    """
    Recreates model from parameters
    """

    # Make pyspeckit be quiet
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        old_log = log.level
        log.setLevel('ERROR')
        # generate a spectrum
        spec = get_spec(scouseobject, spectrum)
        spec.specfit.fittype = bf.fittype
        spec.specfit.fitter = spec.specfit.Registry.multifitters[bf.fittype]
        if bf.ncomps != 0.0:
            mod = np.zeros([len(scouseobject.xtrim), int(bf.ncomps)])
            for k in range(int(bf.ncomps)):
                modparams = bf.params[(k*len(bf.parnames)):(k*len(bf.parnames))+len(bf.parnames)]
                mod[:,k] = spec.specfit.get_model_frompars(scouseobject.xtrim, modparams)
            totmod = np.nansum(mod, axis=1)
            res = (get_flux(scouseobject, spectrum)).value-totmod
        else:
            mod = np.zeros([len(scouseobject.xtrim), 1])
            res = (get_flux(scouseobject, spectrum)).value
        log.setLevel(old_log)

    return mod, res

def get_spec(scouseobject, indiv_spec):
    """
    Generate the spectrum.
    """
    x=scouseobject.xtrim
    y = get_flux(scouseobject, indiv_spec)
    rms=indiv_spec.rms
    spec =  pyspeckit.Spectrum(data=y, error=np.ones(len(y))*rms, xarr=x, \
                              doplot=False, unit=scouseobject.cube.header['BUNIT'],\
                              xarrkwargs={'unit':'km/s'},verbose=False)

    return spec

def check_and_flatten(scouseobject, check_spec_indices, check_block_indices):
    """
    Checks to see if staged_fitting has been initiated. If so it appends
    current list to the existing one.
    """

    _check_spec_indices=None
    _check_block_indices=None

    if np.size(scouseobject.check_spec_indices)!=0:
        _check_spec_indices = [list(scouseobject.check_spec_indices) + list(check_spec_indices)]
        _check_spec_indices = np.unique(_check_spec_indices) # Just in case
    else:
        _check_spec_indices = check_spec_indices

    if np.size(scouseobject.check_block_indices)!=0:
        _check_block_indices = [list(scouseobject.check_block_indices) + list(check_block_indices)]
        _check_block_indices = np.asarray(_check_block_indices)
        _check_block_indices = np.unique(_check_block_indices) # Just in case
    else:
        _check_block_indices = check_block_indices

    return _check_spec_indices, _check_block_indices

def generate_2d_parametermap(scouseobject, spectrum_parameter):
    """
    Create a 2D map of a given spectral parameter
    """
    blankmap = np.zeros(scouseobject.cube.shape[1:])
    blankmap[:] = np.nan

    for ind,spec in ProgressBar(scouseobject.indiv_dict.items()):
        cy,cx = spec.coordinates
        blankmap[cy, cx] = getattr(spec.model, spectrum_parameter)

    return blankmap

def generate_diagnostic_maps(scouseobject, maps=['rms', 'residstd', 'redchi2', 'ncomps', 'aic', 'chi2']):

    return {mapname: generate_2d_parametermap(scouseobject, mapname)
            for mapname in maps}

class DiagnosticImageFigure(object):
    def __init__(self, scouseobject, fig=None, ax=None, keep=False,
                 blocksize=6, mapnames=['rms', 'residstd', 'redchi2', 'ncomps', 'aic', 'chi2'],
                 plotkwargs=dict(interpolation='none', origin='lower'),
                 savedir=None,
                ):
        """
        """

        if fig is None:
            fig = plt.gcf()
        self.fig = fig
        if ax is None:
            ax = plt.gca()
        self.ax = ax
        self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.fig.canvas.mpl_connect('key_press_event', self.keyentry)
        self.blocksize = blocksize
        self.scouseobject = scouseobject
        self.savedir = savedir

        self.mapnames = mapnames
        self.maps = {}

        if savedir is not None:
            loaded = self.load_maps(savedir)
            not_loaded = [x for x in self.mapnames if x not in loaded]
        else:
            not_loaded = self.mapnames

        self.maps.update(generate_diagnostic_maps(self.scouseobject, maps=not_loaded))

        self.save_maps(self.savedir)

        self.plotkwargs = plotkwargs

        self.done = False

        self.done_block_mask = np.zeros_like(self.maps[self.mapnames[0]])
        self.done_con = None

        self.check_spec_indices = []
        self.check_block_indices = []

    def load_maps(self, savedir):
        loaded = []
        for mapname in self.mapnames:
            fn = os.path.join(savedir, "stage_5_"+mapname+".fits")
            if os.path.exists(fn):
                data = fits.getdata(fn)
                self.maps[mapname] = data
                loaded.append(mapname)
        return loaded

    def save_maps(self, savedir, overwrite=True):

        for mapname in self.mapnames:
            fh = fits.PrimaryHDU(data=self.maps[mapname], header=self.scouseobject.cube[0,:,:].header)
            fh.writeto(os.path.join(savedir, "stage_5_"+mapname+".fits"), overwrite=overwrite)

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.click)
        self.fig.canvas.mpl_disconnect(self.keyentry)

    def show_first(self):
        self.ax.imshow(self.maps[self.mapnames[0]], **self.plotkwargs)
        self.ax.set_title('Diagnostic Plot: '+self.mapnames[0]+"\n0:rms; 1:residstd; 2:redchi2; 3:ncomps; 4:aic; 5:chi2")

    def show(self):
        self.fig.canvas.draw()

    def click(self, event):
        """
        What happens following mouse click
        """
        if self.fig.canvas.manager.toolbar._active is None:
            if event.button == 1:

                cx,cy = event.xdata, event.ydata
                if None in (cx,cy):
                    return

                nxblocks, nyblocks, blockarr = get_blocks(self.scouseobject, self.blocksize)
                blockid = blockarr[np.int(cy),np.int(cx)]

                check_spec_indices, check_block_indices=interactive_plot(self.scouseobject, blockrange=[blockid,blockid+1], blocksize=self.blocksize)
                self.check_spec_indices+=check_spec_indices
                self.check_block_indices+=check_block_indices

                self.done_block_mask[(blockarr==blockid)[:self.done_block_mask.shape[0], :self.done_block_mask.shape[1]]] = 1

                if self.done_con is not None:
                    for coll in self.ax.collections:
                        coll.remove()
                self.done_con = self.ax.contourf(self.done_block_mask, colors='w',
                                                 levels=[0.5, 1.5], alpha=0.8)
                print("Number of pixels examined interactively is now {0}".format(self.done_block_mask.sum()))

    def keyentry(self, event):
        if event.key in string.digits and int(event.key) in range(len(self.mapnames)):
            print("Showing map number {0}: {1}".format(event.key, self.mapnames[int(event.key)]))
            self.ax.set_title('Diagnostic Plot: '+self.mapnames[int(event.key)]+"\n0:rms; 1:residstd; 2:redchi2; 3:ncomps; 4:aic; 5:chi2")
            for im in self.ax.images:
                im.remove()
            self.ax.imshow(self.maps[self.mapnames[int(event.key)]],
                           **self.plotkwargs)
            self.show()
        if event.key in ('q', 'enter'):
            self.done = True
