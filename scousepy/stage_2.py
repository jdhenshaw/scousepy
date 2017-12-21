"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2017 Jonathan D. Henshaw
CONTACT: j.d.henshaw[AT]ljmu.ac.uk

"""

import numpy as np
from astropy.io import fits
from astropy import units as u
import sys
from astropy import wcs
import pyspeckit
import warnings
from astropy import log
import matplotlib.pyplot as plt
from .saa_description import add_solution
from .solution_description import fit, print_fit_information

def get_spec(self, x, y, rms):
    """
    Generate the spectrum
    """
    return pyspeckit.Spectrum(data=y, error=np.ones(len(y))*rms, xarr=x, \
                              doplot=True, unit=self.cube.header['BUNIT'],\
                              xarrkwargs={'unit':'km/s'})

def fitting(self, SAA, training_set=False, \
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

            # if no initial guess available then begin by fitting interactively
            if not init_guess:
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

            # else start with an initial guess. If the user isn't happy they
            # can enter the interactive fitting mode
            else:
                spec.specfit(interactive=False, \
                             xmin=self.ppv_vol[0], \
                             xmax=self.ppv_vol[1], guesses=guesses)
                spec.specfit.plot_fit()
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
            happy = h in ['True', 'T', 'true', '1', 't', 'y', 'yes', 'Y', 'Yes', '']
            print("")
            firstgo+=1

        add_solution(SAA, bf)

    else:
        if count==0:
            bf = fitting(self, SAA, \
                         training_set=True, init_guess=False)
        else:
            guesses = self.saa_fits[count-1].params
            bf = fitting(self, SAA, guesses=guesses,\
                         training_set=True, init_guess=True)

    return bf
