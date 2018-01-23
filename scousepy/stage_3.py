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
from astropy import log
from .indiv_spec_description import spectrum
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

                            print("")
                            print(guesses)
                            # Perform fit with parent solution as the initial guess
                            spec.specfit(interactive=False, \
                                         xmin=self.ppv_vol[0], \
                                         xmax=self.ppv_vol[1], \
                                         fittype = model, \
                                         guesses = guesses)

                            # Gen best-fitting solution
                            bf = fit(spec, idx=key, scouse=self)
                            print(bf.params)
                            happy, bf, guesses = check_spec(self, parent_solution, bf)
                            initfit = False


                        sys.exit()
                    log.setLevel(old_log)
                    #sys.exit()

def get_spec(self, x, y, rms):
    """
    Generate the spectrum
    """
    return pyspeckit.Spectrum(data=y, error=np.ones(len(y))*rms, xarr=x, \
                              doplot=True, unit=self.cube.header['BUNIT'],\
                              xarrkwargs={'unit':'km/s'})

def check_spec(self, parent_solution, bf):

    guesses = np.asarray(bf.params)
    condition_passed = np.zeros(4, dtype='bool')

    condition_passed, guesses = check_rms(self, bf, guesses, condition_passed)
    if condition_passed[0]:
        condition_passed, guesses = check_fwhm(self, bf, guesses, condition_passed)
        if condition_passed[1]:
            happy = True
        else:
            happy = False
    else:
        happy = False

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

def check_fwhm(self, bf, guesses, condition_passed):
    """
    Check the fwhm of the best-fitting solution components
    """

    for i in range(int(bf.ncomps)):
        if (bf.params[(i*3)+2] < self.specres*self.tolerances[1]):
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
