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

from .stage_2 import *
from .stage_3 import get_flux, get_indiv_spec, fit_indiv_spectra
from .stage_5 import *

Fitter = Stage2Fitter()
fitting = Fitter.preparefit

from .saa_description import *
from .interactiveplot import showplot
from .solution_description import fit, print_fit_information
from .indiv_spec_description import *

def event_loop():
    fig = plt.gcf()
    while plt.fignum_exists(fig.number):
        try:
            # using just a few little bits of plt.pause below
            plt.gcf().canvas.draw()
            plt.gcf().canvas.start_event_loop(0.1)
            time.sleep(0.1)
        except KeyboardInterrupt:
            break

def check_blocks(scouseobject):
    """
    Checks the current check_spec_indices against those in check_block_indices
    and gets rid of any duplicates
    """
    _check_spec_indices=list(scouseobject.check_spec_indices)
    _check_block_indices=list(scouseobject.check_block_indices)
    _block_indices = []
    # Firstly get the indices associated with the blocks
    nxblocks, nyblocks, blockarr = get_blocks(scouseobject, scouseobject.blocksize)
    spec_mask = pad_spec(scouseobject, scouseobject.blocksize, nxblocks, nyblocks)
    # cycle through the blocks and check the indices against _check_spec_indices
    for blocknum in _check_block_indices:
        keep = (blockarr == blocknum)
        speckeys = spec_mask[keep]
        speckeys = [key for key in speckeys if np.isfinite(key)]
        block_indices = np.array(speckeys)
        sortidx = argsort(block_indices)
        block_indices = block_indices[sortidx]
        _block_indices += list(block_indices)

    # remove any keys from _check_spec_indices that are in _block_indices as they
    # will be fit anyway
    _check_spec_indices = [key for key in _check_spec_indices if not key in _block_indices]

    return _check_spec_indices

def get_block_indices(scouseobject, blocknum):
    """
    Returns a list of indices for the spectra contained within the blocks
    """

    nxblocks, nyblocks, blockarr = get_blocks(scouseobject, scouseobject.blocksize)
    spec_mask = pad_spec(scouseobject, scouseobject.blocksize, nxblocks, nyblocks)

    keep = (blockarr == blocknum)
    speckeys = [int(key) for key in spec_mask[keep] if np.isfinite(key)]
    block_indices = np.sort(speckeys)

    return block_indices

def gen_2d_coords(scouseobject,block_indices):
    """
    Takes flattened indices and returns an array of the 2D indices
    """
    coords=[]
    for idx in block_indices:
        _coords = np.unravel_index(idx, scouseobject.cube.shape[1:])
        coords.append(np.asarray(_coords))
    coords = np.asarray(coords)
    return coords

def gen_pseudo_SAA(scouseobject, coords, block_dict, blocknum, spec):
    """
    Creates an SAA from a list of individual spectra
    """

    # Create spatially averaged spectrum
    for ycrd, xcrd in coords:
        indivspec = scouseobject.cube[:, ycrd, xcrd].value
        spec[:] += indivspec
    spec = spec/len(coords[:,0])
    # Create a pseudo-SAA
    SAA = saa([blocknum,blocknum], spec,
               idx=blocknum, sample=True, scouse=scouseobject)
    block_dict[blocknum] = SAA
    add_ids(SAA, list(coords))

    return SAA

def initialise_indiv_spectra_s6(scouseobject, SAA, njobs):
    """
    Initialise the spectra for fitting
    """
    indiv_spectra = {}
    # Parallel
    if njobs > 1:
        args = [scouseobject, SAA]
        inputs = [[k] + args for k in range(len(SAA.indices_flat))]
        # Send to parallel_map
        indiv_spec = parallel_map(get_indiv_spec, inputs, numcores=njobs)
        merged_spec = [spec for spec in indiv_spec if spec is not None]
        merged_spec = np.asarray(merged_spec)
        for k in range(len(SAA.indices_flat)):
            # Add the spectra to the dict
            key = SAA.indices_flat[k]
            indiv_spectra[key] = merged_spec[k]
    else:
        for k in range(len(SAA.indices_flat)):
            key = SAA.indices_flat[k]
            args = [scouseobject, SAA]
            inputs = [[k] + args]
            inputs = inputs[0]
            indiv_spec = get_indiv_spec(inputs)
            indiv_spectra[key] = indiv_spec
    add_indiv_spectra(SAA, indiv_spectra)

def manually_fit_blocks(scouseobject, block_dict, blocknum):
    """
    Manual fitting of the individual blocks
    """
    # determine how many fits we will actually be performing
    #n_to_fit = sum([block_dict[blocknum].to_be_fit
    #                for blocknum in scouseobject.check_block_indices])

    # Loop through the SAAs
    #for ind, block_ind in enumerate(scouseobject.check_block_indices):
    #    print("Fitting {0} out of {1}".format(ind+1, n_to_fit))
    SAA = block_dict[blocknum]

    with warnings.catch_warnings():
        # This is to catch an annoying matplotlib deprecation warning:
        # "Using default event loop until function specific to this GUI is implemented"
        warnings.simplefilter('ignore', category=DeprecationWarning)

        bf = fitting(scouseobject, SAA, block_dict, blocknum,
                     training_set=False,
                     init_guess=True)

def auto_fit_blocks(scouseobject, block_dict, njobs, blocksize, verbose=False):
    """
    automated fitting of the blocks
    """
    indiv_dictionary = {}
    # Fit the spectra
    fit_indiv_spectra(scouseobject, block_dict, blocksize/3, \
                      njobs=njobs, spatial=False, verbose=verbose, stage=6)

    for block_ind in scouseobject.check_block_indices:
        SAA = block_dict[block_ind]
        for key in SAA.indices_flat:
            spectrum = scouseobject.indiv_dict[key]
            bfmodel = spectrum.model
            alternatives = spectrum.models
            models = []
            models.append([bfmodel])
            models.append(alternatives)

            # Flatten
            models = [mod for mods in models for mod in mods]

            # Now add this as the best-fitting model and add the others to models
            add_bf_model(spectrum, SAA.indiv_spectra[key].model_parent)
            update_model_list(spectrum, models)
            decision = 'block refit'
            add_decision(spectrum, decision)


def get_offsets(radius_pix):
    """
    Returns offsets of adjacent pixels

    Notes:

    For grid size of 3 - returns

    _offsets = np.array([[-1,1], [0,1], [1,1], [-1,0], [0,0], [1,0], [-1,-1], [0,-1], [1,-1]])

    etc.

    """
    arr = np.arange(radius_pix+1)
    sym = np.concatenate((arr*-1,arr )).astype(np.int)
    sym = np.unique(sym)
    _offsets = [pair for pair in itertools.product(sym,sym)]

    return _offsets

def neighbours(n_dim, idx, radius_pix):
    """
    Returns the indices of adjacent pixels
    """

    # Unravel the index of the selected spectrum
    unrav_idx = np.unravel_index(idx, n_dim)

    # Get all the adjacent neighbours
    idxs = [tuple(c) for c in np.add(get_offsets(radius_pix), unrav_idx)]
    idxs = np.array(idxs)

    # Find out which of those neighbours are valid according to the shape of the
    # data cube
    validids = np.full(np.shape(idxs), np.nan)
    valid = (idxs[:,0] >= 0) & (idxs[:,0] < n_dim[0]) & (idxs[:,1] >= 0) & (idxs[:,1] < n_dim[1])
    validids[valid] = idxs[valid,:]
    # Package the valid neighburs up and send them back!
    indices_adjacent = [np.ravel_multi_index(np.array([int(validids[i,0]), int(validids[i,1])]), n_dim) if np.isfinite(validids[i,0]) else np.nan for i in range(len(validids[:,0]))]

    return indices_adjacent

def plot_neighbour_pixels(scouseobject, indices_adjacent, figsize):
    """
    Plot neighbours and their model solutions
    """
    npix = np.size(indices_adjacent)

    # Set up figure page
    fig, axes = pyplot.subplots(int(np.sqrt(npix)), int(np.sqrt(npix)), figsize=figsize)
    fig.canvas.mpl_connect('key_press_event', keyentry)
    fig.patch.set_facecolor('black')
    fig.patch.set_alpha(0.05)
    plt.suptitle("Checking spectrum and its neighbours. Press 'enter' to continue.")
    ax = [a for axis in axes[::-1] for a in axis]

    for i, key in enumerate(indices_adjacent, start=0):

        if np.isfinite(key) and key in scouseobject.indiv_dict:

            spectrum = scouseobject.indiv_dict[key]
            # Get the correct subplot axis
            axis = ax[i]
            # First plot the Spectrum
            axis.plot(scouseobject.xtrim, get_flux(scouseobject, spectrum), 'k-', drawstyle='steps', lw=1)
            # Recreate the model from information held in the solution
            # description
            bfmodel = spectrum.model
            mod, res = recreate_model(scouseobject, spectrum, bfmodel)
            # now overplot the model
            if bfmodel.ncomps == 0.0:
                axis.plot(scouseobject.xtrim, mod[:,0], 'b-', lw=1)
            else:
                for k in range(int(bfmodel.ncomps)):
                    axis.plot(scouseobject.xtrim, mod[:,k], 'b-', lw=1)

            if i != round((npix/2)-0.5):
                axis.patch.set_facecolor('blue')
                axis.patch.set_alpha(0.1)
        else:
            # Get the correct subplot axis
            axis = ax[i]
            axis.plot(0.5, 0.5, 'kx', transform=axis.transAxes, ms=10)
            axis.patch.set_facecolor('blue')
            axis.patch.set_alpha(0.1)

    pyplot.tight_layout(rect=[0, 0.03, 1, 0.95])
    pyplot.show()
    event_loop()

def keyentry(event):
    """
    What happens following a key entry
    """

    if event.key == 'enter':
        plt.close()
        return

def plot_alternatives(scouseobject, key, figsize, plot_residuals=False):
    """
    Plot the spectrum to be checked and its alternatives
    """

    spectrum = scouseobject.indiv_dict[key]
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
        axis.plot(scouseobject.xtrim, get_flux(scouseobject, spectrum), 'k-', drawstyle='steps', lw=1)
        # Recreate the model from information held in the solution
        # description
        mod,res = recreate_model(scouseobject, spectrum, bfmodel)
        # now overplot the model
        if bfmodel.ncomps == 0.0:
            axis.plot(scouseobject.xtrim, mod[:,0], 'b-', lw=1)
        else:
            for k in range(int(bfmodel.ncomps)):
                axis.plot(scouseobject.xtrim, mod[:,k], 'b-', lw=1)
        if plot_residuals:
            axis.plot(scouseobject.xtrim, res,'g-', drawstyle='steps', lw=1)

    # Create the interactive plot
    intplot = showplot(fig, ax, keep=True)
    fig.canvas.mpl_connect('key_press_event', keyentry)
    event_loop()

    return allmodels, intplot.subplots

def update_models(scouseobject, key, models, selection):
    """
    Here we update the model selection based on the users instructions
    """

    spectrum = scouseobject.indiv_dict[key]
    if np.size(selection) == 0.0:
        # If no selection is made - refit manually
        # Make pyspeckit be quiet
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_log = log.level
            log.setLevel('ERROR')
            # generate a spectrum
            spec = get_spec(scouseobject, spectrum)

        log.setLevel(old_log)
        #bf = interactive_fitting(scouseobject, spectrum, spec)
        bf = Stage6Fitter()(scouseobject, spectrum, spec)

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

        print("")
        print("Selection acknowledged. "+
              colors.fg._lightgreen_+"Alternative spectrum selected"+colors._endc_+".")
        print_fit_information(bf, init_guess=False)

    else:
        # If the first spectrum was selected then the user has chosen to accept
        # the current best-fitting solution - so do nothing.
        print("")
        print("Selection acknowledged. "+
              colors.fg._lightgreen_+"Original solution retained"+colors._endc_+".")
        print_fit_information(spectrum.model, init_guess=False)

        pass

class Stage6Fitter(object):
    def __call__(self, *args):
        return self.interactive_fitting(*args)

    def interactive_fitting(self, scouseobject, spectrum, spec):
        """
        Interactive fitter for stage 6
        """
        print("")
        print("Beginning interactive fit of spectrum {0}".format(spectrum))
        self.residuals_shown = False
        self.spec = spec
        self.spectrum = spectrum
        self.scouseobject = scouseobject

        self.happy=False
        while not self.happy:
            # Interactive fitting with pyspeckit
            spec.plotter(xmin=np.min(self.scouseobject.xtrim),
                         xmax=np.max(self.scouseobject.xtrim))
            spec.plotter.figure.canvas.callbacks.disconnect(3)
            spec.specfit.clear_all_connections()
            spec.specfit(interactive=True,
                         print_message=False,
                         xmin=np.min(self.scouseobject.xtrim),
                         xmax=np.max(self.scouseobject.xtrim))

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=DeprecationWarning)

                if plt.matplotlib.rcParams['interactive']:
                    spec.plotter.axis.figure.canvas.mpl_connect('key_press_event',
                                                                self.interactive_callback)
                    event_loop()
                else:
                    plt.show()
                    self.happy = self.interactive_callback('noninteractive')

                if not hasattr(spec.specfit, 'fitter'):
                    raise ValueError("No fitter available for the spectrum."
                                     "  This can occur if you have plt.ion() set"
                                     " or if you did not fit the spectrum."
                                    )

        return self.bf

    def interactive_callback(self, event):
        """
        A 'callback function' to be triggered when the user selects a fit.
        """

        if plt.matplotlib.rcParams['interactive']:
            if hasattr(event, 'key'):

                if event.key in ('enter'):
                    if self.residuals_shown:
                        print("")
                        print("'enter' key acknowledged."+
                        colors.fg._lightgreen_+" Solution accepted"+colors._endc_+".")
                        print("")
                        self.guesses = self.spec.specfit.parinfo.values
                        self.happy = True
                        plt.close(self.spec.plotter.figure.number)
                    return True

                elif event.key == 'esc':
                    self.happy = False
                    self.spec.specfit.clear_all_connections()
                    assert self.spec.plotter._active_gui is None

                elif event.key in ('f', 'F'):
                    print("")
                    print("'f' key acknowledged."+
                    colors.fg._lightred_+" Re-entering interactive fitter"+colors._endc_+".")
                    self.residuals_shown = False

                elif event.key in ('d','D','3',3):
                    # The fit has been performed interactively, but we also
                    # want to print out the nicely-formatted additional
                    # information
                    self.spec.specfit.button3action(event)
                    self.bf = fit(self.spec, idx=self.spectrum.index,
                                  scouse=self.scouseobject)
                    self.spec.specfit.plot_fit(show_components=True)
                    self.spec.specfit.plotresiduals(axis=self.spec.plotter.axis,
                                               clear=False,
                                               color='g',
                                               label=False)
                    self.residuals_shown = True

                    print_fit_information(self.bf, init_guess=True)
                    print("Options:"
                          "\n"
                          "1) If you are happy with this fit, press Enter."
                          "\n"
                          "2) If not, press 'f' to re-enter the interactive fitter.")
                    self.happy = None
                else:
                    self.happy = None

            elif hasattr(event, 'button') and event.button in ('d','D','3',3):
                # The fit has been performed interactively, but we also
                # want to print out the nicely-formatted additional
                # information
                self.bf = fit(self.spec, idx=self.spectrum.index,
                              scouse=self.scouseobject)
                print_fit_information(self.bf, init_guess=True)
                print("Options:"
                      "\n"
                      "1) If you are happy with this fit, press Enter."
                      "\n"
                      "2) If not, press 'f' to re-enter the interactive fitter.")
                self.happy = None
            else:
                self.happy = None
        else:
            # this should only happen if not triggered by a callback
            assert event == 'noninteractive'

            # Best-fitting model solution
            self.bf = fit(self.spec, idx=self.spectrum.index,
                          scouse=self.scouseobject)

            if self.firstgo == 0:
                print("")
                print_fit_information(self.bf, init_guess=True)
                print("")
            else:
                print("")
                print_fit_information(self.bf, init_guess=False)
                print("")

            h = input("Are you happy with the fit? (y/n): ")
            self.happy = h in ['True', 'T', 'true', '1', 't', 'y', 'yes', 'Y', 'Yes']
            print("")
            self.firstgo+=1

            return self.happy
