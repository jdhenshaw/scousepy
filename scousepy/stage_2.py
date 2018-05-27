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

def get_spec(scouseobject, y, rms):
    """
    Generate the spectrum

    (we assume by default that the cube has a rest value defined; if it is not,
    refX will be 0 and km/s <-> frq conversions will break)
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
        """

        if plt.matplotlib.rcParams['interactive']:
            if hasattr(event, 'key'):
                if event.key in ('enter'):
                    if self.residuals_shown:
                        print("'enter' key acknowledged.  Moving to next spectrum "
                              "or next step...")
                        self.happy = True
                        self.bf = fit(self.spec, idx=self.SAA.index,
                                            scouse=self.scouseobject)
                        self.spec.specfit.clear_all_connections()
                        self.spec.plotter.disconnect()
                        assert self.spec.plotter._active_gui is None
                    else:
                        print("'enter' acknowledged.  Guess initialized.  Showing "
                              "fit.")
                        self.firstgo+=1
                        self.guesses = self.spec.specfit.parinfo.values
                        self.trainingset_fit(self.spec,
                                                   init_guess=False,
                                                   guesses=self.guesses,
                                                  )
                elif event.key == 'esc':
                    self.happy = False
                    self.spec.specfit.clear_all_connections()
                    assert self.spec.plotter._active_gui is None
                    self.firstgo+=1
                    self.trainingset_fit(self.spec,
                                               init_guess=True, # re-initialize guess
                                               guesses=self.guesses,
                                              )
                elif event.key in ('f', 'F'):
                    self.residuals_shown = False
                elif event.key in ('d','D','3',3):
                    # The fit has been performed interactively, but we also
                    # want to print out the nicely-formatted additional
                    # information
                    self.spec.specfit.button3action(event)
                    bf = fit(self.spec, idx=self.SAA.index,
                             scouse=self.scouseobject)
                    print_fit_information(bf, init_guess=True)
                    print("If you are happy with this fit, press Enter.  Otherwise, "
                          "use the 'f' key to re-enter the interactive fitter.")
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
                print("If you are happy with this fit, press Enter.  Otherwise, "
                      "use the 'f' key to re-enter the interactive fitter.")
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

    def trainingset_fit(self, spec, init_guess=False, guesses=None):
        self.guesses = guesses
        scouseobject = self.scouseobject
        self.spec = spec

        old_log = log.level
        log.setLevel('ERROR')

        # if this is the initial guess then begin by fitting interactively
        if init_guess:
            self.init_guess = True
            # Interactive fitting with pyspeckit
            spec.plotter(xmin=scouseobject.ppv_vol[0],
                         xmax=scouseobject.ppv_vol[1],
                         figure=plt.figure(1),
                        )
            # disable mpl key commands (especially 'q')
            spec.plotter.figure.canvas.callbacks.disconnect(3)
            spec.specfit.clear_all_connections()
            assert self.spec.plotter._active_gui is None
            spec.specfit(interactive=True,
                         fittype=scouseobject.fittype,
                         print_message=False,
                         xmin=scouseobject.ppv_vol[0],
                         xmax=scouseobject.ppv_vol[1],
                         show_components=True)
            assert self.spec.plotter._active_gui is not None

            self.residuals_shown = False

        # else start with a guess. If the user isn't happy they
        # can enter the interactive fitting mode
        else:
            self.init_guess = False
            spec.plotter(xmin=scouseobject.ppv_vol[0],
                         xmax=scouseobject.ppv_vol[1],
                         figure=plt.figure(1),
                        )
            # disable mpl key commands (especially 'q')
            spec.plotter.figure.canvas.callbacks.disconnect(3)
            spec.specfit.clear_all_connections()
            assert self.spec.plotter._active_gui is None
            spec.specfit(interactive=False,
                         xmin=scouseobject.ppv_vol[0],
                         xmax=scouseobject.ppv_vol[1],
                         guesses=guesses,
                         fittype=scouseobject.fittype)
            spec.specfit.plot_fit(show_components=True)
            spec.specfit.plotresiduals(axis=spec.plotter.axis,
                                       clear=False,
                                       color='g',
                                       label=False)
            assert self.spec.plotter._active_gui is None

            self.residuals_shown = True

        log.setLevel(old_log)


        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)

            if plt.matplotlib.rcParams['interactive']:
                self.happy = None
                spec.plotter.axis.figure.canvas.mpl_connect('key_press_event',
                                                            self.interactive_callback)
                if self.residuals_shown:
                    bf = fit(self.spec, idx=self.SAA.index,
                             scouse=self.scouseobject)
                    print_fit_information(bf, init_guess=True)
                print("If you are happy with this fit, press Enter.  Otherwise, "
                      "use the 'f' key to re-enter the interactive fitter.")
            else:
                plt.show()
                self.happy = self.interactive_callback('noninteractive')

            if not hasattr(spec.specfit, 'fitter'):
                raise ValueError("No fitter available for the spectrum."
                                 "  This can occur if you have plt.ion() set"
                                 " or if you did not fit the spectrum."
                                )

    def fitting(self, scouseobject, SAA, saa_dict, count, training_set=False,
                init_guess=False, guesses=None):

        self.SAA = SAA
        self.scouseobject = scouseobject

        if training_set:
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
            self.trainingset_fit(spec, init_guess=init_guess,
                                       guesses=guesses)

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
            if init_guess:
                bf = self.fitting(scouseobject, SAA, saa_dict, count,
                                        training_set=True,
                                        init_guess=init_guess)
            else:
                model = saa_dict[count].model
                if model is None:
                    bf = self.fitting(scouseobject, SAA, saa_dict, count,
                                            training_set=True, init_guess=True)
                else:
                    guesses = saa_dict[count].model.params
                    bf = self.fitting(scouseobject, SAA, saa_dict,
                                            count, guesses=guesses,
                                            training_set=True,
                                            init_guess=init_guess)

        return bf

def generate_saa_list(scouseobject):
    """
    Returns a list constaining all spectral averaging areas.
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
