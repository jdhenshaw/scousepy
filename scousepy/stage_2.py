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
import time

from astropy.io import fits
from astropy import units as u
from astropy import wcs
from astropy import log

from .saa_description import add_model
from .solution_description import fit, print_fit_information
from .colors import *

def get_spec(scouseobject, y, rms):
    """
    Generate the spectrum. Returns a pyspeckit spectrum object

    (we assume by default that the cube has a rest value defined; if it is not,
    refX will be 0 and km/s <-> frq conversions will break)

    Parameters
    ----------
    scouseobject : Instance of the scousepy class
    y : ndarray
        data corresponding to the spectrum
    rms : ndarray
        rms noise value computed in s1

    """
    fig = plt.figure(1)
    fig.clf()
    ax = fig.gca()
    return pyspeckit.Spectrum(data=y,
                              error=np.ones(len(y))*rms,
                              xarr=scouseobject.xtrim,
                              doplot=True,
                              plotkwargs={'figure': fig, 'axis': ax},
                              unit=scouseobject.cube.header['BUNIT'],
                              xarrkwargs={'unit':'km/s',
                                          'refX': scouseobject.cube.wcs.wcs.restfrq*u.Hz,
                                          # I'm sure there's a way to determine this on the fly...
                                          'velocity_convention': 'radio',
                                         }
                             )


class Stage2Fitter(object):
    def __init__(self, scouseobject=None):
        self.residuals_shown = False
        self.scouseobject = scouseobject

    def interactive_callback(self, event):
        """
        A 'callback function' to be triggered when the user selects a fit.

        Parameters
        ----------
        event : interactive event

        """

        if plt.matplotlib.rcParams['interactive']:
            if hasattr(event, 'key'):

                # Enter to continue
                if event.key in ('enter'):
                    if self.residuals_shown:
                        print("")
                        print("'enter' key acknowledged."+
                        colors.fg._lightgreen_+" Solution accepted"+colors._endc_+".")
                        print("")
                        self.happy = True
                        self.bf = fit(self.spec, idx=self.SAA.index,
                                            scouse=self.scouseobject)
                        self.spec.specfit.clear_all_connections()
                        self.spec.plotter.disconnect()
                        assert self.spec.plotter._active_gui is None
                    else:
                        print("")
                        print("'enter' acknowledged."+
                        colors.fg._cyan_+" Guess initialized, showing fit"+colors._endc_+".")
                        self.firstgo+=1
                        self.guesses = self.spec.specfit.parinfo.values
                        self.scouse_fit(self.spec, init_guess=False,
                                                          guesses=self.guesses,)

                #
                elif event.key == 'esc':
                    self.happy = False
                    self.spec.specfit.clear_all_connections()
                    assert self.spec.plotter._active_gui is None
                    self.firstgo+=1
                    self.scouse_fit(self.spec, init_guess=True,
                                                          guesses=self.guesses,)

                # To re-enter the fitter
                elif event.key in ('f', 'F'):
                    print("")
                    print("'f' key acknowledged."+
                    colors.fg._lightred_+" Re-entering interactive fitter"+colors._endc_+".")
                    self.residuals_shown = False

                # to indicate that all components have been selected
                elif event.key in ('d','D','3',3):
                    # The fit has been performed interactively, but we also
                    # want to print out the nicely-formatted additional
                    # information
                    self.spec.specfit.button3action(event)
                    bf = fit(self.spec, idx=self.SAA.index,
                             scouse=self.scouseobject)
                    print_fit_information(bf, init_guess=self.init_guess)
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
                bf = fit(self.spec, idx=self.SAA.index,
                         scouse=self.scouseobject)
                print_fit_information(bf, init_guess=True)
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
            self.bf = fit(self.spec, idx=self.SAA.index,
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

    def scouse_fit(self, spec, init_guess=False, guesses=None):
        """
        The fitting process followed by scouse

        Parameters:
        spec : pyspeckit spectrum
            Instance of the pyspeckit spectrum class - the spectrum to be fit
        init_guess : bool
            indicates whether the spectrum is the first to be fit (in which case
            the spectrum has to be fit manually)
        guesses : ndarray
            An array of guesses to help the minimisation algorithm converge

        """
        self.guesses = guesses
        scouseobject = self.scouseobject
        self.spec = spec

        old_log = log.level
        log.setLevel('ERROR')

        # The following if statement prepares pyspeckit for fitting

        # if this is the initial guess then begin by fitting interactively
        if init_guess:
            print("")
            print("Press '?' for help with the interactive fitter. ")
            self.init_guess = True
            # Interactive fitting with pyspeckit
            spec.plotter(xmin=np.min(scouseobject.xtrim),
                         xmax=np.max(scouseobject.xtrim),
                         figure=plt.figure(1))
            # disable mpl key commands (especially 'q')
            spec.plotter.figure.canvas.callbacks.disconnect(3)
            spec.specfit.clear_all_connections()
            assert self.spec.plotter._active_gui is None
            spec.specfit(interactive=True,
                         fittype=scouseobject.fittype,
                         print_message=False,
                         xmin=np.min(scouseobject.xtrim),
                         xmax=np.max(scouseobject.xtrim),
                         show_components=True,
                         use_lmfit=True)
            assert self.spec.plotter._active_gui is not None

            self.residuals_shown = False

        # else start with a guess. If the user isn't happy they
        # can enter the interactive fitting mode
        else:
            self.init_guess = False
            spec.plotter(xmin=scouseobject.ppv_vol[0],
                         xmax=scouseobject.ppv_vol[1],
                         figure=plt.figure(1))
            # disable mpl key commands (especially 'q')
            spec.plotter.figure.canvas.callbacks.disconnect(3)
            spec.specfit.clear_all_connections()
            assert self.spec.plotter._active_gui is None

            if None in guesses:
                raise ValueError(colors.fg._red_+"Encountered a 'None' value in"+
                                 " guesses"+colors.fg._endc_)

            spec.specfit(interactive=False,
                         xmin=scouseobject.ppv_vol[0],
                         xmax=scouseobject.ppv_vol[1],
                         guesses=guesses,
                         fittype=scouseobject.fittype,
                         use_lmfit=True)
            spec.specfit.plot_fit(show_components=True)
            spec.specfit.plotresiduals(axis=spec.plotter.axis,
                                       clear=False,
                                       color='g',
                                       label=False)
            assert self.spec.plotter._active_gui is None

            self.residuals_shown = True

        log.setLevel(old_log)

        # Here is where the interactive fitting takes place
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)

            if plt.matplotlib.rcParams['interactive']:
                self.happy = None
                spec.plotter.axis.figure.canvas.mpl_connect('key_press_event',
                                                      self.interactive_callback)
                if self.residuals_shown:
                    bf = fit(self.spec, idx=self.SAA.index,
                             scouse=self.scouseobject)
                    print_fit_information(bf, init_guess=self.init_guess)
                    print("Options:"
                          "\n"
                          "1) If you are happy with this fit, press Enter."
                          "\n"
                          "2) If not, press 'f' to re-enter the interactive fitter.")
            else:
                plt.show()
                self.happy = self.interactive_callback('noninteractive')

            if not hasattr(spec.specfit, 'fitter'):
                raise ValueError(colors.fg._red_+"No fitter available for the "+
                                 "spectrum. This can occur if you have plt.ion()"+
                                 " set or if you did not fit the spectrum."+
                                 colors.fg._endc_)

    def preparefit(self, scouseobject, SAA, saa_dict, count, training_set=False,
                   init_guess=False, guesses=None):

        """
        Preparation for the fitting process followed by scouse - this is called
        in stage 2 of scouse.py.

        The method is broken up into two main processes. The first is the usual
        methodology, whereby we cycle through each spectral averaging area and
        manually fit the spectra. The second is designed for usage where a
        training set is required.

        Parameters
        ----------
        scouseobject : Instance of the scousepy class
        SAA : Instance of the saa class
            contains information regarding the spectral averaging area
        saa_dict : dictionary
            Dictionary housing the SAAs
        count : number
            refers to the index of the previous SAA (unless there isn't one).
            To reduce interactivity scouse will use the previous solution as an
            initial guess to the current spectrum.
        training_set : bool
            indicates whether or not the user is fitting a training set or
            fitting an entire dataset
        init_guess : bool
            indicates whether the spectrum is the first to be fit (in which case
            the spectrum has to be fit manually)
        guesses : ndarray
            An array of guesses to help the minimisation algorithm converge

        """

        self.SAA = SAA
        self.scouseobject = scouseobject

        if training_set:
            # Training set fitting
            # although note that all fits pass through here
            self.happy=False
            self.firstgo = 0

            # Generate the spectrum for pyspeckit to fit
            # Shhh noisy
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                old_log = log.level
                log.setLevel('ERROR')
                spec = get_spec(scouseobject, SAA.ytrim, SAA.rms)
                log.setLevel(old_log)

            self.spec = spec
            self.scouse_fit(spec, init_guess=init_guess, guesses=guesses)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=DeprecationWarning)
                while not self.happy:
                    try:
                        # using just a few little bits of plt.pause below
                        plt.gcf().canvas.draw()
                        plt.gcf().canvas.start_event_loop(0.1)
                        time.sleep(0.1)
                    except KeyboardInterrupt:
                        break

            add_model(SAA, self.bf)
            bf = self.bf

        else:
            # Normal fitting
            if init_guess:
                # if this is the first spectrum don't send any guesses
                bf = self.preparefit(scouseobject, SAA, saa_dict, count,
                                       training_set=True, init_guess=init_guess)
            else:
                # else look for a model
                model = saa_dict[count].model
                if model is None:
                    # If there is no model available for the previous spectrum
                    # use manual fitting
                    bf = self.preparefit(scouseobject, SAA, saa_dict, count,
                                             training_set=True, init_guess=True)
                else:
                    # else send the fitter some guesses
                    guesses = saa_dict[count].model.params
                    bf = self.preparefit(scouseobject, SAA, saa_dict,count,
                        guesses=guesses,training_set=True,init_guess=init_guess)

        return bf

def generate_saa_list(scouseobject):
    """
    Returns a list constaining all spectral averaging areas.

    Parameters
    ----------
    scouseobject : Instance of the scousepy class

    """
    saa_list=[]
    for i in range(len(scouseobject.wsaa)):
        saa_dict = scouseobject.saa_dict[i]
        for j in range(len(saa_dict.keys())):
            # get the relavent SAA
            SAA = saa_dict[j]
            if SAA.to_be_fit:
                saa_list.append([SAA.index, i])

    return saa_list
