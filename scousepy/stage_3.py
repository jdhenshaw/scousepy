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
from .indiv_spec_description import spectrum, add_solution_parent, add_solution_spatial
from .saa_description import add_indiv_spectra
from .solution_description import fit, print_fit_information

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
            if SAA.to_be_fit:
                for k in range(len(SAA.indices_flat)):
                    indiv_spec = spectrum(np.array([SAA.indices[k,0], \
                                          SAA.indices[k,1]]), \
                                          self.cube._data[:,SAA.indices[k,0], \
                                          SAA.indices[k,1]], \
                                          idx=SAA.indices_flat[k], \
                                          scouse=self)
                    indiv_spectra[SAA.indices_flat[k]] = indiv_spec
                    add_indiv_spectra(SAA, indiv_spectra)

def fit_indiv_spectra(self, model = 'gaussian', spatial=False):
    """
    automated fitting procedure for individual spectra
    """
    # Cycle through potentially multiple Rsaa values
    for i in range(len(self.rsaa)):
        # Get the relavent SAA dictionary
        saa_dict = self.saa_dict[i]
        for j in range(len(saa_dict.keys())):
            # get the relavent SAA
            SAA = saa_dict[j]
            # TODO: MANUEL!
            if SAA.to_be_fit:
                # get the parent
                parent_solution = SAA.solution
                # cycle through the spectra contained within this SAA
                for k in range(len(SAA.indices_flat)):
                    # Shhh
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        old_log = log.level
                        log.setLevel('ERROR')

                        # Key to access spectrum in dictionary
                        key = SAA.indices_flat[k]

                        # create the spectrum
                        spec = get_spec(self, SAA.indiv_spectra[key].xtrim, \
                                              SAA.indiv_spectra[key].ytrim, \
                                              SAA.indiv_spectra[key].rms)
                        # Check the solution
                        happy = False
                        initfit = True
                        while not happy:
                            if initfit:
                                guesses = np.asarray(parent_solution.params)

                            if np.sum(guesses) != 0.0:
                                # Perform fit
                                spec.specfit(interactive=False, \
                                            xmin=self.ppv_vol[0], \
                                            xmax=self.ppv_vol[1], \
                                            fittype = model, \
                                            guesses = guesses)

                                # Gen best-fitting solution
                                bf = fit(spec, idx=key, scouse=self)
                                # Check the output model, does it satisfy the
                                # conditions?
                                happy, bf, guesses = check_spec(self, parent_solution, bf, happy)
                                initfit = False
                            else:
                                # If no satisfactory model can be found - fit
                                # a dud!
                                bf = fit(spec, idx=key, scouse=self, fit_dud=True)
                                happy = True

                        # Add the best-fitting solution to the SAA
                        print("")
                        print(SAA.indiv_spectra[key])
                        print(SAA.indiv_spectra[key].solution_parent)
                        add_solution_parent(SAA.indiv_spectra[key], bf)
                        print(SAA.indiv_spectra[key].solution_parent)
                        print_fit_information(bf)
                        print("")
                    log.setLevel(old_log)
                # At this point we have a fit to every spectrum within the SAA.
                # This is where we could implement spatial fitting to complement
                # the fits from the SAA solutions
                if spatial:
                    # TODO: Implement spatial fitting
                    pass


def get_spec(self, x, y, rms):
    """
    Generate the spectrum
    """
    return pyspeckit.Spectrum(data=y, error=np.ones(len(y))*rms, xarr=x, \
                              doplot=True, unit=self.cube.header['BUNIT'],\
                              xarrkwargs={'unit':'km/s'})

def check_spec(self, parent_solution, bf, happy):
    """
    Here we are going to check the output spectrum against user-defined
    tolerance levels described in Henshaw et al. 2016 and against the SAA fit.
    """
    guesses = np.asarray(bf.params)
    condition_passed = np.zeros(3, dtype='bool')
    condition_passed, guesses = check_rms(self, bf, guesses, condition_passed)

    if condition_passed[0]:
        condition_passed, guesses = check_dispersion(self, bf, parent_solution, guesses, condition_passed)
        if (condition_passed[0]) and (condition_passed[1]):
            condition_passed, guesses = check_velocity(self, bf, parent_solution, guesses, condition_passed)
            if np.all(condition_passed) and (bf.ncomps == 1):
                happy = True
            else:
                happy, guesses = check_distinct(self, bf, parent_solution, guesses, happy)

    return happy, bf, guesses

def check_rms(self, bf, guesses, condition_passed):
    """
    Check the rms of the best-fitting solution components
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

def check_dispersion(self, bf, parent_solution, guesses, condition_passed):
    """
    Check the fwhm of the best-fitting solution components
    """

    for i in range(int(bf.ncomps)):

        # Find the closest matching component in the parent SAA solution
        diff = find_closest_match(i, bf, parent_solution)

        # Work out the relative change in velocity dispersion
        idmin = np.squeeze(np.where(diff == np.min(diff)))
        relchange = bf.params[(i*3)+2]/parent_solution.params[(idmin*3)+2]
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

def check_velocity(self, bf, parent_solution, guesses, condition_passed):
    """
    Check the centroid velocity of the best-fitting solution components
    """

    for i in range(int(bf.ncomps)):

        # Find the closest matching component in the parent SAA solution
        diff = find_closest_match(i, bf, parent_solution)

        # Work out the relative change in velocity dispersion
        idmin = np.squeeze(np.where(diff == np.min(diff)))

        # Limits for tolerance
        lower_lim = parent_solution.params[(idmin*3)+1]-(self.tolerances[3]*parent_solution.params[(idmin*3)+2])
        upper_lim = parent_solution.params[(idmin*3)+1]+(self.tolerances[3]*parent_solution.params[(idmin*3)+2])

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

def check_distinct(self, bf, parent_solution, guesses, happy):
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


def find_closest_match(i, bf, parent_solution):
    """
    Find the closest matching component in the parent SAA solution to the current
    component in bf.
    """

    diff = np.zeros(int(parent_solution.ncomps))
    for j in range(int(parent_solution.ncomps)):
        diff[j] = np.sqrt((bf.params[i*3]-parent_solution.params[j*3])**2.+\
                          (bf.params[(i*3)+1]-parent_solution.params[(j*3)+1])**2. + \
                          (bf.params[(i*3)+2]-parent_solution.params[(j*3)+2])**2.)
    return diff
