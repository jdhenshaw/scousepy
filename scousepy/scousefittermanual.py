import numpy as np
import warnings
from astropy import log
import sys
from .colors import *
import matplotlib.pyplot as plt
import time

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

class ScouseFitterManual(object):

    def __init__(self, scousepyfitter):

        """

        """
        self.scousepyfitter=scousepyfitter

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
                        self.happy = True
                        self.spectrum.specfit.clear_all_connections()
                        self.spectrum.plotter.disconnect()
                        plt.close(self.spectrum.plotter.figure.number)
                        assert self.spectrum.plotter._active_gui is None
                    else:
                        print("")
                        print("'enter' key acknowledged."+
                        colors.fg._cyan_+" Showing fit and residuals"+colors._endc_+".")
                        self.fit_updated=True
                        self.guesses = self.spectrum.specfit.parinfo.values
                        self.interactive_fitter()

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
                    self.spectrum.specfit.button3action(event)
                    print("'d' key acknowledged."+
                    colors.fg._cyan_+" Guess initialized"+colors._endc_+".")
                    print('')
                    print("Options:"
                          "\n"
                          "1) To lock the fit and display residuals, press Enter."
                          "\n"
                          "2) Press 'f' to re-enter the interactive fitter.")
                    self.happy = None
                else:
                    self.happy = None

            elif hasattr(event, 'button') and event.button in ('d','D','3',3):
                # The fit has been performed interactively, but we also
                # want to print out the nicely-formatted additional
                # information
                print("'d' key acknowledged."+
                colors.fg._cyan_+" Guess initialized"+colors._endc_+".")
                print('')
                print("Options:"
                      "\n"
                      "1) To lock the fit and display residuals, press Enter."
                      "\n"
                      "2) Press 'f' to re-enter the interactive fitter.")
                self.happy = None
            else:
                self.happy = None
        else:
            # this should only happen if not triggered by a callback
            assert event == 'noninteractive'
            self.printable_format()
            h = input("Are you happy with the fit? (y/n): ")
            self.happy = h in ['True', 'T', 'true', '1', 't', 'y', 'yes', 'Y', 'Yes']
            print("")
            self.fit_updated=True

            return self.happy

    def interactive_fitter(self):
        """
        Interactive fitter - the interactive fitting process controlled by
        manual fit (below). Starts with the interactive fitter with fit_updated
        = false. The user can fit the spectrum. Pressing enter will initialise
        the fit (fit_updated=True). Pressing enter again will accept the fit.
        """

        old_log = log.level
        log.setLevel('ERROR')

        if not self.fit_updated:
            self.fit_updated=False
            # Interactive fitting with pyspeckit
            self.spectrum.plotter(xmin=np.min(self.specx),
                                  xmax=np.max(self.specx),)

            self.spectrum.plotter.figure.canvas.callbacks.disconnect(3)
            self.spectrum.specfit.clear_all_connections()

            assert self.spectrum.plotter._active_gui is None
            # interactive fitting
            self.spectrum.specfit(interactive=True,
                                  print_message=True,
                                  xmin=np.min(self.specx),
                                  xmax=np.max(self.specx),
                                  use_lmfit=True,
                                  show_components=True)
            assert self.spectrum.plotter._active_gui is not None

            self.residuals_shown=False
        else:
            self.fit_updated=True
            self.spectrum.plotter(xmin=np.min(self.specx),
                                  xmax=np.max(self.specx),)
            # disable mpl key commands (especially 'q')
            self.spectrum.plotter.figure.canvas.callbacks.disconnect(3)
            self.spectrum.specfit.clear_all_connections()
            assert self.spectrum.plotter._active_gui is None

            if None in self.guesses:
                raise ValueError(colors.fg._red_+"Encountered a 'None' value in"+
                                 " guesses"+colors._endc_)
            # non interactive - display the fit
            self.spectrum.specfit(interactive=False,
                                  xmin=np.min(self.specx),
                                  xmax=np.max(self.specx),
                                  guesses=self.guesses,
                                  use_lmfit=True)
            self.spectrum.specfit.plot_fit(show_components=True)
            self.spectrum.specfit.plotresiduals(axis=self.spectrum.plotter.axis,
                                       clear=False,
                                       color='g',
                                       label=False)
            assert self.spectrum.plotter._active_gui is None
            self.residuals_shown=True
            self.printable_format()
            print("Options:"
                  "\n"
                  "1) If you are happy with this fit, press Enter."
                  "\n"
                  "2) If not, press 'f' to re-enter the interactive fitter.")


        log.setLevel(old_log)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)

            if plt.matplotlib.rcParams['interactive']:
                self.happy = None
                self.spectrum.plotter.axis.figure.canvas.mpl_connect('key_press_event',self.interactive_callback)
            else:
                plt.show()
                self.happy = self.interactive_callback('noninteractive')

            if not hasattr(self.spectrum.specfit, 'fitter'):
                raise ValueError("No fitter available for the spectrum."
                                 "  This can occur if you have plt.ion() set"
                                 " or if you did not fit the spectrum."
                                )
        return

    def manualfit(self):

        """

        Parameters
        ----------


        """
        plt.ion()
        self.specx=self.scousepyfitter.specx
        self.specy=self.scousepyfitter.specy
        self.specrms=self.scousepyfitter.specrms
        self.spectrum=self.scousepyfitter.spectrum
        self.fit_updated=False
        self.residuals_shown=False
        self.guesses=None
        self.happy=False

        self.interactive_fitter()

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
        plt.ioff()
        return self

    def fit_converge(self):
        if None in self.spectrum.specfit.modelerrs:
            return False
        else:
            return True

    # def get_aic(self):
    #     from astropy.stats import akaike_info_criterion as aic
    #     logl = self.spectrum.specfit.fitter.logp(self.spectrum.xarr, self.spectrum.data, self.spectrum.error)
    #     return aic(logl, int(self.spectrum.specfit.npeaks)+(int(self.spectrum.specfit.npeaks)*3.), len(self.spectrum.xarr))

    def get_aic(self):
        """
        Computes the AIC value
        """
        from astropy.stats import akaike_info_criterion_lsq as aic

        mod = np.zeros([len(self.spectrum.xarr), int(self.spectrum.specfit.npeaks)])
        for k in range(int(self.spectrum.specfit.npeaks)):
            modparams = self.spectrum.specfit.modelpars[(k*len(self.spectrum.specfit.fitter.parnames)):(k*len(self.spectrum.specfit.fitter.parnames))+len(self.spectrum.specfit.fitter.parnames)]
            mod[:,k] = self.spectrum.specfit.get_model_frompars(self.spectrum.xarr, modparams)
        totmod = np.nansum(mod, axis=1)
        res=self.spectrum.data-totmod
        ssr=np.sum((res)**2.0)

        return aic(ssr, (int(self.spectrum.specfit.npeaks)*len(self.spectrum.specfit.fitter.parnames)), len(self.spectrum.xarr))

    def printable_format(self):
        """

        Parameters
        ----------

        """
        specfit=self.spectrum.specfit
        print("")
        print("-----------------------------------------------------")

        print("")
        print("Model type: {0}".format(specfit.fittype))
        print("")
        print(("Number of components: {0}").format(specfit.npeaks))
        print("")
        compcount=0

        if not self.fit_converge():
            print(colors.fg._yellow_+"WARNING: Minimisation failed to converge. Please "
                  "\nrefit manually. "+colors._endc_)
            print("")

        for i in range(0, int(specfit.npeaks)):
            parlow = int((i*len(specfit.fitter.parnames)))
            parhigh = int((i*len(specfit.fitter.parnames))+len(specfit.fitter.parnames))
            parrange = np.arange(parlow,parhigh)
            for j in range(0, len(specfit.fitter.parnames)):
                print(("{0}:  {1} +/- {2}").format(specfit.fitter.parnames[j], \
                                             np.around(specfit.modelpars[parrange[j]],
                                             decimals=5), \
                                             np.around(specfit.modelerrs[parrange[j]],
                                             decimals=5)))
            print("")
            compcount+=1

        print(("chisq:    {0}").format(np.around(specfit.chi2, decimals=2)))
        print(("redchisq: {0}").format(np.around(specfit.chi2/specfit.dof, decimals=2)))
        print(("AIC:      {0}").format(np.around(self.get_aic(), decimals=2)))
        print("-----------------------------------------------------")
        print("")
