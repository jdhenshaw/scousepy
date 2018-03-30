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

def get_spec(self, y, rms):
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
                              xarr=self.xtrim,
                              doplot=True,
                              plotkwargs={'figure': fig, 'axis': ax},
                              unit=self.cube.header['BUNIT'],
                              xarrkwargs={'unit':'km/s',
                                          'refX': self.cube.wcs.wcs.restfrq*u.Hz,
                                          # I'm sure there's a way to determine this on the fly...
                                          'velocity_convention': 'radio',
                                         }
                             )


class Stage2Fitter(object):
    def __init__(thisobject):
        thisobject.residuals_shown = False

    def interactive_callback(thisobject, event):
        """
        A 'callback function' to be triggered when the user selects a fit.
        """

        if plt.matplotlib.rcParams['interactive']:
            if hasattr(event, 'key'):
                if event.key == 'enter':
                    if thisobject.residuals_shown:
                        print("'enter' key acknowledged.  Moving to next spectrum "
                              "or next step...")
                        thisobject.happy = True
                        thisobject.bf = fit(thisobject.spec, idx=thisobject.SAA.index,
                                            scouse=thisobject.self)
                        thisobject.spec.specfit.clear_all_connections()
                        thisobject.spec.plotter.disconnect()
                        assert thisobject.spec.plotter._active_gui is None
                    else:
                        print("'enter' acknowledged.  Guess initialized.  Showing "
                              "fit.")
                        thisobject.firstgo+=1
                        thisobject.guesses = thisobject.spec.specfit.parinfo.values
                        thisobject.trainingset_fit(thisobject.spec,
                                                   init_guess=False,
                                                   guesses=thisobject.guesses,
                                                  )
                elif event.key == 'esc':
                    thisobject.happy = False
                    thisobject.spec.specfit.clear_all_connections()
                    assert thisobject.spec.plotter._active_gui is None
                    thisobject.firstgo+=1
                    thisobject.trainingset_fit(thisobject.spec,
                                               init_guess=True, # re-initialize guess
                                               guesses=thisobject.guesses,
                                              )
                elif event.key in ('f', 'F'):
                    thisobject.residuals_shown = False
                elif event.key in ('d','D','3',3):
                    # The fit has been performed interactively, but we also
                    # want to print out the nicely-formatted additional
                    # information
                    bf = fit(thisobject.spec, idx=thisobject.SAA.index,
                             scouse=thisobject.self)
                    print_fit_information(bf, init_guess=True)
                    print("If you are happy with this fit, press Enter.  Otherwise, "
                          "use the 'f' key to re-enter the interactive fitter.")
                    thisobject.happy = None
                else:
                    thisobject.happy = None
            elif hasattr(event, 'button') and event.button in ('d','D','3',3):
                # The fit has been performed interactively, but we also
                # want to print out the nicely-formatted additional
                # information
                bf = fit(thisobject.spec, idx=thisobject.SAA.index,
                         scouse=thisobject.self)
                print_fit_information(bf, init_guess=True)
                print("If you are happy with this fit, press Enter.  Otherwise, "
                      "use the 'f' key to re-enter the interactive fitter.")
                thisobject.happy = None
            else:
                thisobject.happy = None
        else:
            # this should only happen if not triggered by a callback
            assert event == 'noninteractive'

            # Best-fitting model solution
            thisobject.bf = fit(thisobject.spec, idx=thisobject.SAA.index,
                                scouse=thisobject.self)

            if thisobject.firstgo == 0:
                print("")
                print_fit_information(thisobject.bf, init_guess=True)
                print("")
            else:
                print("")
                print_fit_information(thisobject.bf, init_guess=False)
                print("")

            h = input("Are you happy with the fit? (y/n): ")
            thisobject.happy = h in ['True', 'T', 'true', '1', 't', 'y', 'yes', 'Y', 'Yes']
            print("")
            thisobject.firstgo+=1

            return thisobject.happy

    def trainingset_fit(thisobject, spec, init_guess=False, guesses=None):
        thisobject.guesses = guesses
        self = thisobject.self

        old_log = log.level
        log.setLevel('ERROR')

        # if this is the initial guess then begin by fitting interactively
        if init_guess:
            thisobject.init_guess = True
            # Interactive fitting with pyspeckit
            spec.plotter(xmin=self.ppv_vol[0],
                         xmax=self.ppv_vol[1],
                         figure=plt.figure(1),
                        )
            spec.specfit.clear_all_connections()
            assert thisobject.spec.plotter._active_gui is None
            spec.specfit(interactive=True,
                         fittype=self.fittype,
                         print_message=False,
                         xmin=self.ppv_vol[0],
                         xmax=self.ppv_vol[1],
                         show_components=True)
            assert thisobject.spec.plotter._active_gui is not None

            thisobject.residuals_shown = False

        # else start with a guess. If the user isn't happy they
        # can enter the interactive fitting mode
        else:
            thisobject.init_guess = False
            spec.plotter(xmin=self.ppv_vol[0],
                         xmax=self.ppv_vol[1],
                         figure=plt.figure(1),
                        )
            spec.specfit.clear_all_connections()
            assert thisobject.spec.plotter._active_gui is None
            spec.specfit(interactive=False,
                         xmin=self.ppv_vol[0],
                         xmax=self.ppv_vol[1],
                         guesses=guesses,
                         fittype=self.fittype)
            spec.specfit.plot_fit(show_components=True)
            spec.specfit.plotresiduals(axis=spec.plotter.axis,
                                       clear=False,
                                       color='g',
                                       label=False)
            assert thisobject.spec.plotter._active_gui is None

            thisobject.residuals_shown = True

        log.setLevel(old_log)


        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)

            if plt.matplotlib.rcParams['interactive']:
                thisobject.happy = None
                spec.plotter.axis.figure.canvas.mpl_connect('key_press_event',
                                                            thisobject.interactive_callback)
                if thisobject.residuals_shown:
                    bf = fit(thisobject.spec, idx=thisobject.SAA.index,
                             scouse=thisobject.self)
                    print_fit_information(bf, init_guess=True)
                print("If you are happy with this fit, press Enter.  Otherwise, "
                      "use the 'f' key to re-enter the interactive fitter.")
            else:
                plt.show()
                thisobject.happy = thisobject.interactive_callback('noninteractive')

            if not hasattr(spec.specfit, 'fitter'):
                raise ValueError("No fitter available for the spectrum."
                                 "  This can occur if you have plt.ion() set"
                                 " or if you did not fit the spectrum."
                                )

    def fitting(thisobject, self, SAA, saa_dict, count, training_set=False,
                init_guess=False, guesses=None):

        thisobject.SAA = SAA
        thisobject.self = self

        if training_set:
            thisobject.happy=False
            thisobject.firstgo = 0

            # Generate the spectrum for pyspeckit to fit
            # Shhh noisy
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                old_log = log.level
                log.setLevel('ERROR')
                spec = get_spec(self, SAA.ytrim, SAA.rms)
                log.setLevel(old_log)

            thisobject.spec = spec
            thisobject.trainingset_fit(spec, init_guess=init_guess,
                                       guesses=guesses)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=DeprecationWarning)
                while not thisobject.happy:
                    try:
                        plt.pause(0.1)
                    except KeyboardInterrupt:
                        break

            add_model(SAA, thisobject.bf)
            bf = thisobject.bf

        else:
            if init_guess:
                bf = thisobject.fitting(self, SAA, saa_dict, count,
                                        training_set=True,
                                        init_guess=init_guess)
            else:
                model = saa_dict[count].model
                if model is None:
                    bf = thisobject.fitting(self, SAA, saa_dict, count,
                                            training_set=True, init_guess=True)
                else:
                    guesses = saa_dict[count].model.params
                    bf = thisobject.fitting(self, SAA, saa_dict,
                                            count, guesses=guesses,
                                            training_set=True,
                                            init_guess=init_guess)

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
            if SAA.to_be_fit:
                saa_list.append([SAA.index, i])

    return saa_list
