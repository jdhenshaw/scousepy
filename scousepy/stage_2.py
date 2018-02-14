# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np
import pyspeckit
import matplotlib.pyplot as plt
import sys
import warnings

from astropy.io import fits
from astropy import units as u
from astropy import wcs
from astropy import log

from .saa_description import add_model
from .solution_description import fit, print_fit_information

def get_spec(self, x, y, rms):
    """
    Generate the spectrum
    """
    return pyspeckit.Spectrum(data=y, error=np.ones(len(y))*rms, xarr=x, \
                              doplot=True, unit=self.cube.header['BUNIT'],\
                              xarrkwargs={'unit':'km/s'})

def fitting(self, SAA, saa_dict, count, training_set=False, \
            init_guess=False, guesses=None):

    if training_set:
        happy=False
        firstgo = 0
        while not happy:
            spec=None
            bf=None
            # Generate the spectrum for pyspeckit to fit
            # Shhh noisy
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                old_log = log.level
                log.setLevel('ERROR')
                spec = get_spec(self, SAA.xtrim, SAA.ytrim, SAA.rms)
                log.setLevel(old_log)

            # if this is the initial guess then begin by fitting interactively
            if init_guess:
                # Interactive fitting with pyspeckit
                spec.plotter(xmin=self.ppv_vol[0], \
                             xmax=self.ppv_vol[1])
                spec.specfit(interactive=True, \
                             xmin=self.ppv_vol[0], \
                             xmax=self.ppv_vol[1])
                plt.show()
                # Best-fitting model solution
                bf = fit(spec, idx=SAA.index, scouse=self)

                print("")
                print_fit_information(bf, init_guess=False)
                print("")

            # else start with a guess. If the user isn't happy they
            # can enter the interactive fitting mode
            else:
                spec.plotter(xmin=self.ppv_vol[0], \
                             xmax=self.ppv_vol[1])
                spec.specfit(interactive=False, \
                             xmin=self.ppv_vol[0], \
                             xmax=self.ppv_vol[1], \
                             guesses=guesses,\
                             fittype = self.fittype)
                spec.specfit.plot_fit(show_components=True)
                spec.specfit.plotresiduals(axis=spec.plotter.axis,clear=False,color='g',label=False)

                plt.show()
                bf = fit(spec, idx=SAA.index, scouse=self)

                if firstgo == 0:
                    print("")
                    print_fit_information(bf, init_guess=True)
                    print("")
                else:
                    print("")
                    print_fit_information(bf, init_guess=False)
                    print("")

            h = input("Are you happy with the fit? (y/n): ")
            happy = h in ['True', 'T', 'true', '1', 't', 'y', 'yes', 'Y', 'Yes']
            print("")
            firstgo+=1
        add_model(SAA, bf)

    else:
        if init_guess:
            bf = fitting(self, SAA, saa_dict, count, \
                         training_set=True, init_guess=init_guess)
        else:
            guesses = saa_dict[count].model.params
            bf = fitting(self, SAA, saa_dict, count, guesses=guesses,\
                         training_set=True, init_guess=init_guess)

    return bf

def generate_saa_list(self):
    """
    Returns a list constaining all spectral averaging areas.
    """
    saa_list=[]
    for i in range(len(self.rsaa)):
        saa_dict = self.saa_dict[i]
        for j in range(len(saa_dict.keys())):
            # get the relavent SAA
            SAA = saa_dict[j]
            saa_list.append([SAA.index, i])

    return saa_list
