# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np
import sys
import warnings
import pyspeckit
import matplotlib.pyplot as plt
import itertools
from astropy import log

from .indiv_spec_description import *
from .parallel_map import *
from .saa_description import add_indiv_spectra, clean_up, merge_models
from .solution_description import fit, print_fit_information
from .verbose_output import print_to_terminal


def initialise_indiv_spectra(self):
    """
    Here, the individual spectra are primed ready for fitting. We create a new
    object for each spectrum and they are contained within a dictionary which
    can be located within the relavent SAA.
    """
    # Cycle through potentially multiple Rsaa values
    for i in range(len(self.rsaa)):
        # Get the relavent SAA dictionary
        saa_dict = self.saa_dict[i]
        for j in range(len(saa_dict.keys())):
            # get the relavent SAA
            SAA = saa_dict[j]
            # Initialise indiv spectra
            indiv_spectra = {}

            for k in range(len(SAA.indices_flat)):
                indiv_spec = spectrum(np.array([SAA.indices[k,0], \
                                      SAA.indices[k,1]]), \
                                      self.cube._data[:,SAA.indices[k,0], \
                                      SAA.indices[k,1]], \
                                      idx=SAA.indices_flat[k], \
                                      scouse=self)
                indiv_spectra[SAA.indices_flat[k]] = indiv_spec
            add_indiv_spectra(SAA, indiv_spectra)


def fit_indiv_spectra(self, saa_dict, rsaa, njobs=1, \
                      spatial=False, verbose=False):
    """
    Automated fitting procedure for individual spectra
    """

    if verbose:
        count=0
        progress_bar = print_to_terminal(stage='s3', step='fitting', length=len(saa_dict.keys()), var=rsaa)

    for j in range(len(saa_dict.keys())):
        if verbose:
            progress_bar + 1
            progress_bar.show_progress()

        # get the relavent SAA
        SAA = saa_dict[j]

        if SAA.to_be_fit:
            # Get the SAA model solution
            parent_model = SAA.model

            # Fitting process is parallelised
            if njobs > 1:
                args = [self, SAA, parent_model]
                inputs = [[k] + args for k in range(len(SAA.indices_flat))]
                # Send to parallel_map
                bf = parallel_map(fit_spec, inputs, numcores=njobs)
                merged_bfs = [core_bf for core_bf in bf if core_bf is not None]
                merged_bfs = np.asarray(merged_bfs)
                for k in range(len(SAA.indices_flat)):
                    # Add the models to the spectra
                    key = SAA.indices_flat[k]
                    add_model_parent(SAA.indiv_spectra[key], merged_bfs[k,0])
                    add_model_dud(SAA.indiv_spectra[key], merged_bfs[k,1])
            else:
                # If njobs = 1 just cycle through
                for k in range(len(SAA.indices_flat)):
                    key = SAA.indices_flat[k]
                    args = [self, SAA, parent_model]
                    inputs = [[k] + args]
                    inputs = inputs[0]
                    bfs = fit_spec(inputs)
                    add_model_parent(SAA.indiv_spectra[key], bfs[0])
                    add_model_dud(SAA.indiv_spectra[key], bfs[1])
        else:
            # Fitting duds
            if njobs > 1:
                if len(SAA.indices_flat) != 0:
                    args = [self, SAA]
                    inputs = [[k] + args for k in range(len(SAA.indices_flat))]
                    dud = parallel_map(fit_dud, inputs, numcores=njobs)
                    merged_duds = [core_bf for core_bf in dud if core_bf is not None]
                    merged_duds = np.asarray(merged_duds)
                    for k in range(len(SAA.indices_flat)):
                        key = SAA.indices_flat[k]
                        add_model_parent(SAA.indiv_spectra[key], merged_duds[k])
            else:
                for k in range(len(SAA.indices_flat)):
                    key = SAA.indices_flat[k]
                    args = [self, SAA]
                    inputs = [[k] + args]
                    inputs = inputs[0]
                    dud = fit_dud(inputs)
                    add_model_parent(SAA.indiv_spectra[key], dud)

def get_spec(self, x, y, rms):
    """
    Generate the spectrum.
    """
    return pyspeckit.Spectrum(data=y, error=np.ones(len(y))*rms, xarr=x, \
                              doplot=False, unit=self.cube.header['BUNIT'],\
                              xarrkwargs={'unit':'km/s'},verbose=False)

def fit_spec(inputs):
    """
    Process used for fitting spectra. Returns a best-fit solution and a dud for
    every spectrum.
    """
    idx, self, SAA, parent_model = inputs
    key = SAA.indices_flat[idx]
    spec=None
    # Shhh
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        old_log = log.level
        log.setLevel('ERROR')

        # create the spectrum
        spec = get_spec(self, SAA.indiv_spectra[key].xtrim, \
                              SAA.indiv_spectra[key].ytrim, \
                              SAA.indiv_spectra[key].rms)
        log.setLevel(old_log)

    bf = fitting_process_parent(self, SAA, key, spec, parent_model)
    dud = fitting_process_duds(self, SAA, key, None)
    return [bf, dud]

def fit_dud(inputs):
    """
    Directly fitting duds. Kept separate so that we don't have to generate a
    spectrum every time.
    """
    idx, self, SAA = inputs
    # Key to access spectrum in dictionary
    key = SAA.indices_flat[idx]
    spec = None
    # Fit a dud
    bf = fitting_process_duds(self, SAA, key, spec)
    return bf

def fitting_process_parent(self, SAA, key, spec, parent_model):
    """
    The process used for fitting individual spectra using the arent SAA solution
    """
    # Check the model
    happy = False
    initfit = True
    while not happy:
        if initfit:
            guesses = np.asarray(parent_model.params)
        if np.sum(guesses) != 0.0:
            # Perform fit
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                old_log = log.level
                log.setLevel('ERROR')

                spec.specfit(interactive=False, \
                             clear_all_connections=True,\
                             xmin=self.ppv_vol[0], \
                             xmax=self.ppv_vol[1], \
                             fittype = self.model, \
                             guesses = guesses,\
                             verbose=False)

                log.setLevel(old_log)
            # Gen best-fitting model
            bf = fit(spec, idx=key, scouse=self)

            # Check the output model, does it satisfy the
            # conditions?
            happy, bf, guesses = check_spec(self, parent_model, bf, happy)
            initfit = False
        else:
            # If no satisfactory model can be found - fit a dud!
            bf = fitting_process_duds(self, SAA, key, None)

            happy = True

    return bf

def fitting_process_duds(self, SAA, key, spec):
    """
    Fitting duds
    """
    bf = fit(spec, idx=key, scouse=self, fit_dud=True,\
             noise=SAA.indiv_spectra[key].rms, \
             duddata=SAA.indiv_spectra[key].ytrim)

    return bf

def check_spec(self, parent_model, bf, happy):
    """
    Here we are going to check the output spectrum against user-defined
    tolerance levels described in Henshaw et al. 2016 and against the SAA fit.
    """

    guesses = np.asarray(bf.params)
    condition_passed = np.zeros(3, dtype='bool')
    condition_passed, guesses = check_rms(self, bf, guesses, condition_passed)
    if condition_passed[0]:
        condition_passed, guesses = check_dispersion(self, bf, parent_model, guesses, condition_passed)
        if (condition_passed[0]) and (condition_passed[1]):
            condition_passed, guesses = check_velocity(self, bf, parent_model, guesses, condition_passed)
            if np.all(condition_passed):
                if (bf.ncomps == 1):
                    happy = True
                else:
                    happy, guesses = check_distinct(self, bf, parent_model, guesses, happy)

    return happy, bf, guesses

def check_rms(self, bf, guesses, condition_passed):
    """
    Check the rms of the best-fitting model components
    """

    for i in range(int(bf.ncomps)):
        if (bf.params[i*3] < bf.rms*self.tolerances[0]) or (bf.params[i*3] < bf.errors[i*3]*self.tolerances[0]):
            guesses[i*3] = 0.0
            guesses[(i*3)+1] = 0.0
            guesses[(i*3)+2] = 0.0

    violating_comps = (guesses==0.0)
    if np.any(violating_comps):
        condition_passed[0]=False
    else:
        condition_passed[0]=True

    guesses = guesses[(guesses != 0.0)]

    return condition_passed, guesses

def check_dispersion(self, bf, parent_model, guesses, condition_passed):
    """
    Check the fwhm of the best-fitting model components
    """

    for i in range(int(bf.ncomps)):

        # Find the closest matching component in the parent SAA model
        diff = find_closest_match(i, bf, parent_model)

        # Work out the relative change in velocity dispersion
        idmin = np.squeeze(np.where(diff == np.min(diff)))
        relchange = bf.params[(i*3)+2]/parent_model.params[(idmin*3)+2]
        if relchange < 1.:
            relchange = 1./relchange

        # Does this satisfy the criteria
        if (bf.params[(i*3)+2] < self.specres*self.tolerances[1]) or (relchange > self.tolerances[2]):
            guesses[i*3] = 0.0
            guesses[(i*3)+1] = 0.0
            guesses[(i*3)+2] = 0.0

    violating_comps = (guesses==0.0)
    if np.any(violating_comps):
        condition_passed[1]=False
    else:
        condition_passed[1]=True

    guesses = guesses[(guesses != 0.0)]

    return condition_passed, guesses

def check_velocity(self, bf, parent_model, guesses, condition_passed):
    """
    Check the centroid velocity of the best-fitting model components
    """

    for i in range(int(bf.ncomps)):

        # Find the closest matching component in the parent SAA model
        diff = find_closest_match(i, bf, parent_model)

        # Work out the relative change in velocity dispersion
        idmin = np.squeeze(np.where(diff == np.min(diff)))

        # Limits for tolerance
        lower_lim = parent_model.params[(idmin*3)+1]-(self.tolerances[3]*parent_model.params[(idmin*3)+2])
        upper_lim = parent_model.params[(idmin*3)+1]+(self.tolerances[3]*parent_model.params[(idmin*3)+2])

        # Does this satisfy the criteria
        if (bf.params[(i*3)+1] < lower_lim) or (bf.params[(i*3)+1] > upper_lim):
            guesses[i*3] = 0.0
            guesses[(i*3)+1] = 0.0
            guesses[(i*3)+2] = 0.0

    violating_comps = (guesses==0.0)
    if np.any(violating_comps):
        condition_passed[2]=False
    else:
        condition_passed[2]=True

    guesses = guesses[(guesses != 0.0)]

    return condition_passed, guesses

def check_distinct(self, bf, parent_model, guesses, happy):
    """
    Check to see if component pairs can be distinguished
    """

    fwhmconv = 2.*np.sqrt(2.*np.log(2.))

    intlist  = [bf.params[(i*3)] for i in range(int(bf.ncomps))]
    velolist = [bf.params[(i*3)+1] for i in range(int(bf.ncomps))]
    displist = [bf.params[(i*3)+2] for i in range(int(bf.ncomps))]

    diff = np.zeros(int(bf.ncomps))
    validvs = np.ones(int(bf.ncomps))

    for i in range(int(bf.ncomps)):

        if validvs[i] != 0.0:

            for j in range(int(bf.ncomps)):
                diff[j] = abs(velolist[i]-velolist[j])
            diff[(diff==0.0)] = np.nan

            idmin = np.squeeze(np.where(diff==np.nanmin(diff)))

            adjacent_intensity = intlist[idmin]
            adjacent_velocity = velolist[idmin]
            adjacent_dispersion = displist[idmin]

            sep = np.abs(velolist[i] - adjacent_velocity)
            min_allowed_sep = np.min(np.array([displist[i], adjacent_dispersion]))*fwhmconv

            if sep > min_allowed_sep:
                if validvs[idmin] !=0.0:
                    validvs[i] = 1.0
                    validvs[idmin] = 1.0
                else:
                    validvs[i] = 1.0
                    validvs[idmin] = 0.0

                    intlist[idmin] = 0.0
                    velolist[idmin] = 0.0
                    displist[idmin] = 0.0
            else:
                # If the components do not satisfy the criteria then average
                # them and use the new quantities as input guesses
                validvs[i] = 1.0
                validvs[idmin] = 0.0

                intlist[i] = np.mean([intlist[i], adjacent_intensity])
                velolist[i] = np.mean([velolist[i], adjacent_velocity])
                displist[i] = np.mean([displist[i], adjacent_dispersion])

                intlist[idmin] = 0.0
                velolist[idmin] = 0.0
                displist[idmin] = 0.0

    for i in range(int(bf.ncomps)):
        guesses[(i*3)] = intlist[i]
        guesses[(i*3)+1] = velolist[i]
        guesses[(i*3)+2] = displist[i]

    violating_comps = (guesses==0.0)
    if np.any(violating_comps):
        happy=False
    else:
        happy=True

    guesses = guesses[(guesses != 0.0)]

    return happy, guesses


def find_closest_match(i, bf, parent_model):
    """
    Find the closest matching component in the parent SAA model to the current
    component in bf.
    """

    diff = np.zeros(int(parent_model.ncomps))
    for j in range(int(parent_model.ncomps)):
        diff[j] = np.sqrt((bf.params[i*3]-parent_model.params[j*3])**2.+\
                          (bf.params[(i*3)+1]-parent_model.params[(j*3)+1])**2. + \
                          (bf.params[(i*3)+2]-parent_model.params[(j*3)+2])**2.)
    return diff

def compile_spectra(self, saa_dict, indiv_dict, rsaa, spatial=False, verbose=False):
    """
    Here we compile all best-fitting models into a single dictionary.
    """

    if verbose:
        progress_bar = print_to_terminal(stage='s3', step='compile', length=0, var=rsaa)

    key_list = []
    model_list = []

    for j in range(len(saa_dict.keys())):
        # get the relavent SAA
        SAA = saa_dict[j]
        #if SAA.to_be_fit:
            # Get indiv dict
        indiv_spectra = SAA.indiv_spectra
        if np.size(indiv_spectra) != 0:
            for key in indiv_spectra.keys():
                key_list.append(key)
                model_list.append(indiv_spectra[key])

    # sort the lists
    key_arr = np.array(key_list)
    model_arr = np.array(model_list)
    sortidx = argsort(key_list)
    key_arr = key_arr[sortidx]
    model_arr = model_arr[sortidx]

    # Cycle through all the spectra
    for key in range(self.cube.shape[1]*self.cube.shape[2]):

        # Find all instances of key in the key_arr
        model_idxs = np.squeeze(np.where(key_arr == key))
        # If there is a solution available
        if np.size(model_idxs) > 0:
            # If there is only one instance of this spectrum being fit - we can
            # add it to the dictionary straight away
            if np.size(model_idxs) == 1:
                _spectrum = model_arr[model_idxs]
                model_list = []
                model_list = get_model_list(model_list, _spectrum, spatial)
                update_model_list(_spectrum, model_list)
                indiv_dict[key] = _spectrum
            else:
                # if not, we have to compile the solutions into a single object
                # Take the first one
                _spectrum = model_arr[model_idxs[0]]
                model_list = []
                model_list = get_model_list(model_list, _spectrum, spatial)
                # Now cycle through the others
                for i in range(1, np.size(model_idxs)):
                    _spec = model_arr[model_idxs[i]]
                    model_list = get_model_list(model_list, _spec, spatial)
                # So now the model list should contain every single model
                # solution that is available from all spectral averaging areas

                # Update the model list of the first spectrum and then update
                # the dictionary
                update_model_list(_spectrum, model_list)
                indiv_dict[key] = _spectrum

    # this is the complete list of all spectra included in all dictionaries
    key_set = set(key_arr)
    key_set = list(key_set)

    return key_set

def compile_key_sets(self, key_set):
    """
    Returns unqiue keys
    """
    if len(self.rsaa) == 1:
        key_set=key_set[0]
        self.key_set = key_set
    else:
        key_set = [key for keys in key_set for key in keys]
        key_set = set(key_set)
        self.key_set = list(key_set)

def merge_dictionaries(self, indiv_dictionaries, spatial=False, verbose=False):
    """
    There is now a dictionary for each Rsaa - merge these into a single one
    """

    if verbose:
        progress_bar = print_to_terminal(stage='s3', step='merge', length=0)

    main_dict={}
    if len(self.rsaa)>1:
        for key in self.key_set:
            # Search dictionaries for found keys
            keyfound = np.zeros(len(self.rsaa), dtype='bool')
            for i in range(len(indiv_dictionaries.keys())):
                if key in indiv_dictionaries[i]:
                    keyfound[i] = True

            # If the key is found take the first spectrum and add to the new
            # dictionary
            idx = list(np.squeeze(np.where(keyfound==True)))
            if np.size(idx)==1:
                main_dict[key] = indiv_dictionaries[idx][key]
                idx.remove(idx)
            else:
                main_dict[key] = indiv_dictionaries[idx[0]][key]
                idx.remove(idx[0])

            # Get the main spectrum
            main_spectrum = main_dict[key]
            # Merge spectra from other dictionaries into main_dict

            if np.size(idx) != 0:
                for i in idx:
                    dictionary = indiv_dictionaries[i]
                    _spectrum = dictionary[key]
                    merge_models(main_spectrum, _spectrum)

        # Return this new dictionary
        self.indiv_dict = main_dict
    else:
        main_dictionary = indiv_dictionaries[0]
        # Return this new dictionary
        self.indiv_dict = main_dictionary

def remove_duplicates(self, verbose):
    """
    Removes duplicate models from the model dictionary
    """
    if verbose:
        progress_bar = print_to_terminal(stage='s3', step='duplicates', length=0)

    for key in self.indiv_dict.keys():
        # get the spectrum
        _spectrum = self.indiv_dict[key]
        # get the models
        models = _spectrum.models

        # extract aic values and identify unique values
        aiclist = []
        for i in range(len(models)):
            aiclist.append(models[i].aic)
        aicarr = np.asarray(aiclist)
        aicarr = np.around(aicarr, decimals=5)
        uniqvals, uniqids = np.unique(aicarr, return_index=True)
        models = np.asarray(models)
        uniqmodels = models[uniqids]

        # update list with only unique aic entries
        uniqmodels = list(uniqmodels)
        update_model_list_remdup(_spectrum, uniqmodels)


def get_model_list(model_list, _spectrum, spatial=False):
    """
    Add to model list
    """

    model_list.append(_spectrum.model_parent)

    if _spectrum.model_dud is not None:
        model_list.append(_spectrum.model_dud)
    if spatial:
        model_list.append(_spectrum.model_spatial)

    return model_list

def argsort(data, reversed=False):
    """
    Returns sorted indices

    Notes
    -----
    Sorting in Python 2. and Python 3. can differ in cases where you have
    identical values.

    """

    index = np.arange(len(data))
    key = lambda x: data[x]
    sortidx = sorted(index, key=key,reverse=reversed)
    sortidx = np.array(list(sortidx))
    return sortidx

def clean_SAAs(self, saa_dict):
    """
    This is to save space - there is lots of (often duplicated) information
    stored within the SAAs - get rid of this.
    """

    for j in range(len(saa_dict.keys())):
        # get the relavent SAA
        SAA = saa_dict[j]
        if SAA.to_be_fit:
            clean_up(SAA)
