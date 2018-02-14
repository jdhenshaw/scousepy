# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
import sys
from matplotlib import pyplot
from matplotlib.patches import Rectangle

from .stage_5 import *

from .interactiveplot import showplot
from .solution_description import fit, print_fit_information
from .indiv_spec_description import *

def get_offsets(radius_pix):
    """
    Returns offsets of adjacent pixels

    Notes:

    For grid size of 3 - returns

    _offsets = np.array([[-1,1], [0,1], [1,1], [-1,0], [0,0], [1,0], [-1,-1], [0,-1], [1,-1]])

    etc.

    """
    arr = np.arange(radius_pix+1)
    sym = np.concatenate((arr,arr * -1)).astype(np.int)
    sym = np.unique(sym)

    _offsets = [pair for pair in itertools.product(sym,sym)]

    return _offsets

def neighbours(n_dim, idx, radius_pix):
    """
    Returns the indices of adjacent pixels
    """

    # Unravel the index of the selected spectrum
    unrav_idx = np.unravel_index(idx, np.flip(n_dim,0))

    # Get all the adjacent neighbours
    idxs = [tuple(c) for c in np.add(get_offsets(radius_pix), unrav_idx)]
    idxs = np.array(idxs)

    # Find out which of those neighbours are valid according to the shape of the
    # data cube
    validids = np.full(np.shape(idxs), np.nan)
    valid = (idxs[:,0] >= 0) & (idxs[:,0] < n_dim[1]) & (idxs[:,1] >= 0) & (idxs[:,1] < n_dim[0])
    validids[valid] = idxs[valid,:]

    # Package the valid neighburs up and send them back!
    indices_adjacent = [np.ravel_multi_index(np.array([int(validids[i,0]), int(validids[i,1])]), np.flip(n_dim, 0)) if np.isfinite(validids[i,0]) else np.nan for i in range(len(validids[:,0]))]

    return indices_adjacent

def plot_neighbour_pixels(self, indices_adjacent, figsize):
    """
    Plot neighbours and their model solutions
    """
    npix = np.size(indices_adjacent)

    # Set up figure page
    fig, ax = pyplot.subplots(int(np.sqrt(npix)), int(np.sqrt(npix)), figsize=figsize)
    fig.canvas.mpl_connect('key_press_event', keyentry)
    fig.patch.set_facecolor('black')
    fig.patch.set_alpha(0.05)
    plt.suptitle("Checking spectrum and its neighbours. Press 'enter' to continue.")
    ax = np.asarray(ax)
    ax = ax.T
    ax = np.flip(ax,1)
    ax = [a for axis in ax for a in axis]

    for i, key in enumerate(indices_adjacent, start=0):

        if np.isfinite(key):

            spectrum = self.indiv_dict[key]
            # Get the correct subplot axis
            axis = ax[i]
            # First plot the Spectrum
            axis.plot(spectrum.xtrim, spectrum.ytrim, 'k-', drawstyle='steps', lw=1)
            # Recreate the model from information held in the solution
            # description
            bfmodel = spectrum.model
            mod = recreate_model(self, spectrum, bfmodel)
            # now overplot the model
            if bfmodel.ncomps == 0.0:
                axis.plot(spectrum.xtrim, mod[:,0], 'b-', lw=1)
            else:
                for k in range(int(bfmodel.ncomps)):
                    axis.plot(spectrum.xtrim, mod[:,k], 'b-', lw=1)

            if i != round((npix/2)-0.5):
                axis.patch.set_facecolor('blue')
                axis.patch.set_alpha(0.1)
        else:
            # Get the correct subplot axis
            axis = ax[i]
            axis.plot(0.5, 0.5, 'kX', transform=axis.transAxes, ms=10)
            axis.patch.set_facecolor('blue')
            axis.patch.set_alpha(0.1)

    pyplot.tight_layout(rect=[0, 0.03, 1, 0.95])
    pyplot.show()

def keyentry(event):
    """
    What happens following a key entry
    """

    if event.key == 'enter':
        plt.close()
        return

def plot_alternatives(self, key, figsize, plot_residuals=False):
    """
    Plot the spectrum to be checked and its alternatives
    """

    spectrum = self.indiv_dict[key]
    bfmodel = spectrum.model
    alternatives = spectrum.models

    allmodels = []
    allmodels.append([bfmodel])
    allmodels.append(alternatives)

    # Flatten
    allmodels = [mod for mods in allmodels for mod in mods]

    lenmods = np.size(allmodels)

    # Set up figure page
    fig, ax = pyplot.subplots(1, lenmods, figsize=[12,5])

    for i in range(lenmods):
        # Get the correct subplot axis
        if lenmods == 1:
            axis = ax
        else:
            axis = ax[i]

        bfmodel = allmodels[i]

        # First plot the Spectrum
        axis.plot(spectrum.xtrim, spectrum.ytrim, 'k-', drawstyle='steps', lw=1)
        # Recreate the model from information held in the solution
        # description
        mod = recreate_model(self, spectrum, bfmodel)
        # now overplot the model
        if bfmodel.ncomps == 0.0:
            axis.plot(spectrum.xtrim, mod[:,0], 'b-', lw=1)
        else:
            for k in range(int(bfmodel.ncomps)):
                axis.plot(spectrum.xtrim, mod[:,k], 'b-', lw=1)
        if plot_residuals:
            axis.plot(spectrum.xtrim, bfmodel.residuals,'g-', drawstyle='steps', lw=1)

    # Create the interactive plot
    intplot = showplot(fig, ax, keep=True)

    return allmodels, intplot.subplots

def update_models(self, key, models, selection):
    """
    Here we update the model selection based on the users instructions
    """

    spectrum = self.indiv_dict[key]
    if np.size(selection) == 0.0:
        # If no selection is made - refit manually
        # Make pyspeckit be quiet
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_log = log.level
            log.setLevel('ERROR')
            # generate a spectrum
            spec = get_spec(self, \
                            spectrum.xtrim, \
                            spectrum.ytrim, \
                            spectrum.rms)
        log.setLevel(old_log)
        bf = interactive_fitting(self, spectrum, spec)

        # Now add this as the best-fitting model and add the others to models
        add_bf_model(spectrum, bf)
        update_model_list(spectrum, models)

        decision = 'refit'
        add_decision(spectrum, decision)

    elif selection[0] != 0.0:
        # If any spectrum other than the first is selected then swap this to the
        # model and the current best fit to models
        bf = models[selection[0]]
        models.remove(models[selection[0]])
        # Now add this as the best-fitting model and add the others to models
        add_bf_model(spectrum, bf)
        update_model_list(spectrum, models)

        decision = 'alternative'
        add_decision(spectrum, decision)
        
    else:
        # If the first spectrum was selected then the user has chosen to accept
        # the current best-fitting solution - so do nothing.
        pass

def interactive_fitting(self, spectrum, spec):
    """
    Interactive fitter for stage 6
    """
    happy=False
    while not happy:
        bf=None

        # Interactive fitting with pyspeckit
        spec.plotter(xmin=self.ppv_vol[0], \
                     xmax=self.ppv_vol[1])
        spec.specfit(interactive=True, \
                     xmin=self.ppv_vol[0], \
                     xmax=self.ppv_vol[1])
        plt.show()

        # Best-fitting model solution
        bf = fit(spec, idx=spectrum.index, scouse=self)

        print("")
        print_fit_information(bf, init_guess=False)
        print("")

        h = input("Are you happy with the fit? (y/n): ")
        happy = h in ['True', 'T', 'true', '1', 't', 'y', 'yes', 'Y', 'Yes']
        print("")

    return bf
