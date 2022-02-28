# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""
import numpy as np
from .colors import *
import time
from astropy import log
import warnings
import matplotlib.pyplot as plt

class Decomposer(object):
    """
    A class containing various methods for decomposing individual spectra

    Parameters
    ----------
    spectral_axis : array
        An array of the spectral axis
    spectrum : array
        The spectrum
    rms : number
        An estimate of the rms

    Attributes
    ----------
    fittype : string
        A string describing the pyspeckit fitter
    guesses : list
        A list containing the initial guesses to the fit
    guesses_updated : list
        Used if the best-fitting solution is to be compared with a parent
        spectrum (as in scouse)
    psktemplate : instance of pyspeckit's Spectrum class
        A template spectrum generated using pyspeckit
    pskspectrum : instance of pyspeckit's Spectrum class
        This is the spectrum that will be fit
    modeldict : dictionary
        A dictionary describing the best-fitting solution
    validfit : bool
        Whether or not the fit is valid. Used if the best-fitting solution is to
        be compared with a parent spectrum (as in scouse)
    tol : list
        list of tolerance values used to compare the best-fitting solution to
        that of its parent spectrum
    res : number
        the channel spacing
    method : string
        The fitting method used. Current options:

            parent: the fitting method predominantly used by scouse, where a
                    spectrum has been fit using initial guesses from a parent
                    spectrum
            dspec:  Where a spectrum has been fit using input guesses from
                    derivative spectroscopy
            manual: Where a spectrum has been fit manually using pyspeckit's
                    interactive fitter

    """
    def __init__(self,spectral_axis,spectrum,rms):

        self.spectral_axis=spectral_axis
        self.spectrum=spectrum
        self.rms=rms
        self.fittype=None
        self.guesses=None
        self.guesses_from_parent=None
        self.guesses_updated=None
        self.psktemplate=None
        self.pskspectrum=None
        self.modeldict=None
        self.validfit=False
        self.tol=None
        self.res=None
        self.method=None
        self.fit_updated=False
        self.residuals_shown=False
        self.guesses=None
        self.happy=False
        self.conditions=None

    def fit_spectrum_with_guesses(self, guesses, fittype='gaussian'):
        """
        Fitting method used when using scouse as a standalone fitter. It takes
        guesses supplied by dspec and calls on pyspeckit to fit the spectrum

        Parameters
        ----------
        guesses : list
            a list containing the initial guesses for the fit parameters
        fittype : string
            A string describing the pyspeckit fitter

        """
        self.method='dspec'
        self.fittype=fittype
        self.guesses=guesses

        self.fit_a_spectrum()
        self.get_model_information()

    def fit_spectrum_from_parent(self,guesses,guesses_parent,tol,res,fittype='gaussian'):
        """
        The fitting method most commonly used by scouse. This method will fit
        a spectrum and compare the result against another model. Most commonly
        a model describing a lower resolution parent spectrum

        Parameters
        ----------
        guesses : list
            a list containing the initial guesses for the fit parameters
        guesses_parent : list
            a list containing the model parameters of the parent
        tol : list
            list of tolerance values used to compare the best-fitting solution to
            that of its parent spectrum
        res : number
            the channel spacing
        fittype : string
            A string describing the pyspeckit fitter
        """
        self.method='parent'
        self.fittype=fittype
        self.guesses=guesses
        self.guesses_parent=guesses_parent
        self.tol=tol
        self.res=res

        if self.psktemplate is not None:
            self.update_template()
        else:
            self.create_a_spectrum()
        self.fit_a_spectrum()

        errors=np.copy(self.pskspectrum.specfit.modelerrs)
        errors=[np.nan if error is None else error for error in errors ]
        errors=np.asarray([np.nan if np.invert(np.isfinite(error)) else error for error in errors  ])

        if np.any(np.invert(np.isfinite(errors))):
            print('initial fit did not converge...modifying initial guesses')
            guesses = np.copy(self.pskspectrum.specfit.modelpars)
            rounding = np.asarray([np.abs(np.floor(np.log10(guess))) if np.floor(np.log10(guess))<0.0 else 1.0 for guess in guesses])
            print(rounding)
            self.guesses = np.asarray([np.around(guess,decimals=int(rounding[i])) for i, guess in enumerate(guesses)])

            nparams=np.size(self.pskspectrum.specfit.fitter.parnames)

            ncomponents=np.size(self.guesses)/nparams

            for i in range(int(ncomponents)):
                component = self.guesses[int((i*nparams)):int((i*nparams)+nparams)]
                if np.sum([1 for number in component if number < 0.0]) >= 1:
                    self.guesses[int((i*nparams)):int((i*nparams)+nparams)] = 0.0


            namelist = ['tex', 'amp', 'amplitude', 'peak', 'tant', 'tmb']
            foundname = [pname in namelist for pname in self.pskspectrum.specfit.fitter.parnames]
            foundname = np.array(foundname)
            idx=np.where(foundname==True)[0]
            idx=np.asscalar(idx[0])

            # Now check all components to see if they are above the rms threshold
            amplist=np.asarray([self.guesses[int(i*nparams)+idx] for i in range(int(ncomponents))])

            idx = np.where(amplist==np.min(amplist))
            idx=np.asscalar(idx[0])

            self.guesses[int((idx*nparams)):int((idx*nparams)+nparams)] = 0.0

            self.psktemplate=None
            self.pskspectrum=None
            if self.psktemplate is not None:
                self.update_template()
            else:
                self.create_a_spectrum()

            self.guesses = self.guesses[(self.guesses != 0.0)]
            self.fit_a_spectrum()

        self.get_model_information()
        self.check_against_parent()
        if not self.validfit:
            self.modeldict={}
        self.psktemplate=None
        self.pskspectrum=None

    def fit_spectrum_manually(self, fittype='gaussian'):
        """
        Method used to manually fit a spectrum

        Parameters
        ----------
        fittype : string
            A string describing the pyspeckit fitter
        """
        plt.ion()
        self.method='manual'
        self.fittype=fittype

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

        self.get_model_information()

    def interactive_fitter(self):
        """
        Interactive fitter - the interactive fitting process controlled by
        fit_spectrum_manually. Starts with the interactive fitter with
        fit_updated= false. The user can fit the spectrum. Pressing enter will
        initialise the fit (fit_updated=True). Pressing enter again will
        accept the fit.
        """
        old_log = log.level
        log.setLevel('ERROR')

        if not self.fit_updated:
            self.fit_updated=False
            # Interactive fitting with pyspeckit
            self.pskspectrum.plotter(xmin=np.min(self.spectral_axis),
                                    xmax=np.max(self.spectral_axis),)

            self.pskspectrum.plotter.figure.canvas.callbacks.disconnect(3)
            self.pskspectrum.specfit.clear_all_connections()

            assert self.pskspectrum.plotter._active_gui is None
            # interactive fitting
            self.fit_a_spectrum_interactively()
            assert self.pskspectrum.plotter._active_gui is not None
            self.residuals_shown=False
        else:
            self.fit_updated=True
            self.pskspectrum.plotter(xmin=np.min(self.spectral_axis),
                                    xmax=np.max(self.spectral_axis),)
            # disable mpl key commands (especially 'q')
            self.pskspectrum.plotter.figure.canvas.callbacks.disconnect(3)
            self.pskspectrum.specfit.clear_all_connections()
            assert self.pskspectrum.plotter._active_gui is None

            if None in self.guesses:
                raise ValueError(colors.fg._red_+"Encountered a 'None' value in"+
                                 " guesses"+colors._endc_)
            # non interactive - display the fit
            self.fit_a_spectrum()
            self.pskspectrum.specfit.plot_fit(show_components=True)
            self.pskspectrum.specfit.plotresiduals(axis=self.pskspectrum.plotter.axis,
                                       clear=False,
                                       color='g',
                                       label=False)
            assert self.pskspectrum.plotter._active_gui is None
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
                self.pskspectrum.plotter.axis.figure.canvas.mpl_connect('key_press_event',self.interactive_callback)
            else:
                plt.show()
                self.happy = self.interactive_callback('noninteractive')
            #
            if not hasattr(self.pskspectrum.specfit, 'fitter'):
                raise ValueError("No fitter available for the spectrum."
                                 "  This can occur if you have plt.ion() set"
                                 " or if you did not fit the spectrum."
                                )
        return

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
                        self.pskspectrum.specfit.clear_all_connections()
                        self.pskspectrum.plotter.disconnect()
                        plt.close(self.pskspectrum.plotter.figure.number)
                        assert self.pskspectrum.plotter._active_gui is None
                    else:
                        print("")
                        print("'enter' key acknowledged."+
                        colors.fg._cyan_+" Showing fit and residuals"+colors._endc_+".")
                        self.fit_updated=True
                        self.guesses = self.pskspectrum.specfit.parinfo.values
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
                    self.pskspectrum.specfit.button3action(event)
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

    def fit_a_spectrum(self):
        """
        Fits a spectrum

        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_log = log.level
            log.setLevel('ERROR')
            self.pskspectrum.specfit(interactive=False,
                                clear_all_connections=True,
                                xmin=np.min(self.spectral_axis),
                                xmax=np.max(self.spectral_axis),
                                fittype = self.fittype,
                                guesses = self.guesses,
                                verbose=False,
                                use_lmfit=True)
            log.setLevel(old_log)

    def fit_a_spectrum_interactively(self):
        """
        Fits a spectrum interactively

        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_log = log.level
            log.setLevel('ERROR')
            self.pskspectrum.specfit(interactive=True,
                                print_message=True,
                                xmin=np.min(self.spectral_axis),
                                xmax=np.max(self.spectral_axis),
                                fittype = self.fittype,
                                verbose=False,
                                use_lmfit=True,
                                show_components=True)
            log.setLevel(old_log)

    def create_a_template(self,unit='',xarrkwargs={}):
        """
        generates an instance of pyspeckit's Spectrum class

        Parameters
        ----------
        x : array
            spectral axis
        y : array
            the spectrum
        rms : number
            estimate of the rms
        unit : str
            unit of the spectral axis
        xarrkwargs : dictionary
            key word arguments describing the spectral axis
        """
        from pyspeckit import Spectrum

        spectrum=np.zeros_like(self.spectral_axis,dtype='float')
        error_spectrum=np.ones_like(self.spectral_axis,dtype='float')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_log = log.level
            log.setLevel('ERROR')
            self.psktemplate = Spectrum(data=spectrum,
                                        error=error_spectrum,
                                        xarr=self.spectral_axis,
                                        doplot=False,
                                        unit=unit,
                                        xarrkwargs=xarrkwargs,
                                        verbose=False,
                                        )

            log.setLevel(old_log)

    def create_a_spectrum(self,unit='',xarrkwargs={}):
        """
        generates an instance of pyspeckit's Spectrum class

        Parameters
        ----------
        x : array
            spectral axis
        y : array
            the spectrum
        rms : number
            estimate of the rms
        unit : str
            unit of the spectral axis
        xarrkwargs : dictionary
            key word arguments describing the spectral axis
        """
        from pyspeckit import Spectrum

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_log = log.level
            log.setLevel('ERROR')
            self.pskspectrum = Spectrum(data=self.spectrum,
                                        error=np.ones_like(self.spectrum)*self.rms,
                                        xarr=self.spectral_axis,
                                        doplot=False,
                                        unit=unit,
                                        xarrkwargs=xarrkwargs,
                                        verbose=False,
                                        )

            log.setLevel(old_log)

    def update_template(self):
        """
        updates a template spectrum with the spectrum values

        """
        import astropy.units as u
        import copy
        # create a copy of the template
        self.pskspectrum=copy.copy(self.psktemplate)
        # update important values
        self.pskspectrum.specfit.Spectrum = self.pskspectrum
        self.pskspectrum.data = u.Quantity(self.spectrum).value
        self.pskspectrum.error = u.Quantity(np.ones_like(self.spectrum)*self.rms).value
        self.pskspectrum.specfit.spectofit = u.Quantity(self.spectrum).value
        self.pskspectrum.specfit.errspec = u.Quantity(np.ones_like(self.spectrum)*self.rms).value

    def get_model_information(self):
        """
        Framework for model solution dictionary
        """
        self.modeldict={}

        if (None in self.pskspectrum.specfit.modelerrs):
            self.modeldict['fittype']=None
            self.modeldict['parnames']=self.pskspectrum.specfit.fitter.parnames
            self.modeldict['ncomps']=0
            self.modeldict['params']=np.zeros(len(self.pskspectrum.specfit.modelerrs))
            self.modeldict['errors']=np.zeros(len(self.pskspectrum.specfit.modelerrs))
            if np.ma.is_masked(self.pskspectrum.error):
                idnonmasked=np.where(~self.pskspectrum.error.mask)[0]
                if np.size(idnonmasked)==0:
                    self.modeldict['rms']=np.nan
                else:
                    self.modeldict['rms']=self.pskspectrum.error[idnonmasked[0]]
            else:
                self.modeldict['rms']=self.pskspectrum.error[0]
            self.modeldict['residstd']= np.std(self.pskspectrum.data)
            self.modeldict['chisq']=0.0
            self.modeldict['dof']=0.0
            self.modeldict['redchisq']=0.0
            self.modeldict['AIC']=0.0
            self.modeldict['fitconverge'] = self.fit_converge()
            self.modeldict['method']=self.method
        else:
            self.modeldict['fittype']=self.pskspectrum.specfit.fittype
            self.modeldict['parnames']=self.pskspectrum.specfit.fitter.parnames
            self.modeldict['ncomps']=int(self.pskspectrum.specfit.npeaks)
            self.modeldict['params']=self.pskspectrum.specfit.modelpars
            self.modeldict['errors']=self.pskspectrum.specfit.modelerrs
            if np.ma.is_masked(self.pskspectrum.error):
                idnonmasked=np.where(~self.pskspectrum.error.mask)[0]
                if np.size(idnonmasked)==0:
                    self.modeldict['rms']=np.nan
                else:
                    self.modeldict['rms']=self.pskspectrum.error[idnonmasked[0]]
            else:
                self.modeldict['rms']=self.pskspectrum.error[0]
            self.modeldict['residstd']= np.std(self.pskspectrum.specfit.residuals)
            self.modeldict['chisq']=self.pskspectrum.specfit.chi2
            self.modeldict['dof']=self.pskspectrum.specfit.dof
            self.modeldict['redchisq']=self.pskspectrum.specfit.chi2/self.pskspectrum.specfit.dof
            self.modeldict['AIC']=self.get_aic()
            self.modeldict['fitconverge'] = self.fit_converge()
            self.modeldict['method']=self.method

    def get_aic(self):
        """
        Computes the AIC value
        """
        from astropy.stats import akaike_info_criterion_lsq as aic

        mod = np.zeros([len(self.pskspectrum.xarr), int(self.pskspectrum.specfit.npeaks)])
        for k in range(int(self.pskspectrum.specfit.npeaks)):
            modparams = self.pskspectrum.specfit.modelpars[(k*len(self.pskspectrum.specfit.fitter.parnames)):(k*len(self.pskspectrum.specfit.fitter.parnames))+len(self.pskspectrum.specfit.fitter.parnames)]
            mod[:,k] = self.pskspectrum.specfit.get_model_frompars(self.pskspectrum.xarr, modparams)
        totmod = np.nansum(mod, axis=1)
        res=self.pskspectrum.data-totmod
        ssr=np.sum((res)**2.0)

        return aic(ssr, (int(self.pskspectrum.specfit.npeaks)*len(self.pskspectrum.specfit.fitter.parnames)), len(self.pskspectrum.xarr))

    def check_against_parent(self):
        """

        """
        self.guesses_updated=np.asarray(self.modeldict['params'])
        condition_passed = np.zeros(5, dtype='bool')

        condition_passed = self.check_ncomps(condition_passed)

        if condition_passed[0]:
            condition_passed=self.check_finite(condition_passed)
            if (condition_passed[0]) and (condition_passed[1]):
                condition_passed=self.check_rms(condition_passed)
                if (condition_passed[0]) and (condition_passed[1]) and (condition_passed[2]):
                    condition_passed=self.check_dispersion(condition_passed)
                    if (condition_passed[0]) and (condition_passed[1]) and (condition_passed[2]) and (condition_passed[3]):
                        condition_passed=self.check_velocity(condition_passed)
                        if np.all(condition_passed):
                            if int((np.size(self.guesses_updated)/np.size(self.modeldict['parnames']))==1):
                                self.validfit = True
                            else:
                                self.check_distinct()

        self.conditions=condition_passed

    def check_ncomps(self, condition_passed):
        """
        Check to see if the number of components in the fit has changed beyond
        a reasonable amount

        """
        nparams=np.size(self.modeldict['parnames'])
        ncomponents_parent=np.size(self.guesses_parent)/nparams
        ncomponents_child=np.size(self.guesses_updated)/nparams

        ncompdiff = np.abs(ncomponents_parent-ncomponents_child)

        if ncompdiff > self.tol[0]:
            condition_passed[0]=False
            self.guesses_updated=[]
        else:
            condition_passed[0]=True

        return condition_passed

    def check_finite(self, condition_passed):
        """
        Check to see if the number of components in the fit are not finite

        """

        nparams=np.size(self.modeldict['parnames'])
        ncomponents=np.size(self.guesses_updated)/nparams

        # Now check all components to see if they are finite
        for i in range(int(ncomponents)):
            # find violating components
            if np.any(np.isnan(self.guesses_updated[int(i*nparams):int(i*nparams)+nparams])):
                self.guesses_updated[int((i*nparams)):int((i*nparams)+nparams)] = 0.0

        violating_comps = (self.guesses_updated==0.0)
        if np.any(violating_comps):
            condition_passed[1]=False
        else:
            condition_passed[1]=True

        self.guesses_updated = self.guesses_updated[(self.guesses_updated != 0.0)]

        return condition_passed

    def check_rms(self,condition_passed):
        """
        Check the rms of the best-fitting model components

        Parameters
        ----------
        condition_passed : list
            boolean list indicating which quality control steps have been satisfied

        """
        # Find where in the parameter array the "amplitude" is located. Make this
        # general to allow for other models
        namelist = ['tex', 'amp', 'amplitude', 'peak', 'tant', 'tmb']
        foundname = [pname in namelist for pname in self.modeldict['parnames']]
        foundname = np.array(foundname)
        idx=np.where(foundname==True)[0]
        idx=np.asscalar(idx[0])

        nparams=np.size(self.modeldict['parnames'])
        ncomponents=np.size(self.guesses_updated)/nparams

        # Now check all components to see if they are above the rms threshold
        for i in range(int(ncomponents)):
            if (self.guesses_updated[int(i*nparams)+idx] < self.rms*self.tol[1]):
                self.guesses_updated[int((i*nparams)):int((i*nparams)+nparams)] = 0.0

        violating_comps = (self.guesses_updated==0.0)
        if np.any(violating_comps):
            condition_passed[2]=False
        else:
            condition_passed[2]=True

        self.guesses_updated = self.guesses_updated[(self.guesses_updated != 0.0)]

        return condition_passed

    def check_dispersion(self,condition_passed):
        """
        Check the fwhm of the best-fitting model components

        Parameters
        ----------
        condition_passed : list
            boolean list indicating which quality control steps have been satisfied

        """

        fwhmconv = 2.*np.sqrt(2.*np.log(2.))

        # Find where the velocity dispersion is located in the parameter array
        namelist = ['dispersion', 'width', 'fwhm']
        foundname = [pname in namelist for pname in self.modeldict['parnames']]
        foundname = np.array(foundname)
        idx=np.where(foundname==True)[0]
        idx=np.asscalar(idx[0])

        nparams=np.size(self.modeldict['parnames'])
        ncomponents=np.size(self.guesses_updated)/nparams

        for i in range(int(ncomponents)):

            # Find the closest matching component in the parent SAA model
            diff = self.find_closest_match(i, nparams)
            idmin = np.where(diff == np.min(diff))[0]
            idmin = idmin[0]

            # Work out the relative change in velocity dispersion
            relchange = self.guesses_updated[int((i*nparams)+idx)]/self.guesses_parent[int((idmin*nparams)+idx)]
            if relchange < 1.:
                relchange = 1./relchange

            # Does this satisfy the criteria
            if (self.guesses_updated[int((i*nparams)+idx)]*fwhmconv < self.res*self.tol[2]) or \
               (relchange > self.tol[3]):
                # set to zero
                self.guesses_updated[int((i*nparams)):int((i*nparams)+nparams)] = 0.0

        violating_comps = (self.guesses_updated==0.0)
        if np.any(violating_comps):
            condition_passed[3]=False
        else:
            condition_passed[3]=True

        self.guesses_updated = self.guesses_updated[(self.guesses_updated != 0.0)]

        return condition_passed

    def check_velocity(self,condition_passed):
        """
        Check the centroid velocity of the best-fitting model components

        Parameters
        ----------
        condition_passed : list
            boolean list indicating which quality control steps have been satisfied

        """

        # Find where the peak is located in the parameter array
        namelist = ['velocity', 'shift', 'centroid', 'center']
        foundname = [pname in namelist for pname in self.modeldict['parnames']]
        foundname = np.array(foundname)
        idxv=np.where(foundname==True)[0]
        idxv=np.asscalar(idxv[0])

        # Find where the velocity dispersion is located in the parameter array
        namelist = ['dispersion', 'width', 'fwhm']
        foundname = [pname in namelist for pname in self.modeldict['parnames']]
        foundname = np.array(foundname)
        idxd=np.where(foundname==True)[0]
        idxd=np.asscalar(idxd[0])

        nparams=np.size(self.modeldict['parnames'])
        ncomponents=np.size(self.guesses_updated)/nparams

        for i in range(int(ncomponents)):

            # Find the closest matching component in the parent SAA model
            diff = self.find_closest_match(i, nparams)
            idmin = np.where(diff == np.min(diff))[0]
            idmin = idmin[0]

            # Limits for tolerance
            lower_lim = self.guesses_parent[int((idmin*nparams)+idxv)]-(self.tol[4]*self.guesses_parent[int((idmin*nparams)+idxd)])
            upper_lim = self.guesses_parent[int((idmin*nparams)+idxv)]+(self.tol[4]*self.guesses_parent[int((idmin*nparams)+idxd)])
            # Does this satisfy the criteria
            if (self.guesses_updated[(i*nparams)+idxv] < lower_lim) or \
               (self.guesses_updated[(i*nparams)+idxv] > upper_lim):
                # set to zero
                self.guesses_updated[int((i*nparams)):int((i*nparams)+nparams)] = 0.0

        violating_comps = (self.guesses_updated==0.0)
        if np.any(violating_comps):
            condition_passed[4]=False
        else:
            condition_passed[4]=True

        self.guesses_updated = self.guesses_updated[(self.guesses_updated != 0.0)]

        return condition_passed

    def check_distinct(self):
        """
        Check to see if component pairs can be distinguished in velocity

        """

        # Find where the peak is located in the parameter array
        namelist = ['tex', 'amp', 'amplitude', 'peak', 'tant', 'tmb']
        foundname = [pname in namelist for pname in self.modeldict['parnames']]
        foundname = np.array(foundname)
        idxp=np.where(foundname==True)[0]
        idxp=np.asscalar(idxp[0])

        # Find where the peak is located in the parameter array
        namelist = ['velocity', 'shift', 'centroid', 'center']
        foundname = [pname in namelist for pname in self.modeldict['parnames']]
        foundname = np.array(foundname)
        idxv=np.where(foundname==True)[0]
        idxv=np.asscalar(idxv[0])

        # Find where the velocity dispersion is located in the parameter array
        namelist = ['dispersion', 'width', 'fwhm']
        foundname = [pname in namelist for pname in self.modeldict['parnames']]
        foundname = np.array(foundname)
        idxd=np.where(foundname==True)[0]
        idxd=np.asscalar(idxd[0])

        fwhmconv = 2.*np.sqrt(2.*np.log(2.))

        nparams=np.size(self.modeldict['parnames'])
        ncomponents=np.size(self.guesses_updated)/nparams

        intlist  = [self.guesses_updated[int((i*nparams)+idxp)] for i in range(int(ncomponents))]
        velolist = [self.guesses_updated[int((i*nparams)+idxv)] for i in range(int(ncomponents))]
        displist = [self.guesses_updated[int((i*nparams)+idxd)] for i in range(int(ncomponents))]

        diff = np.zeros(int(ncomponents))
        validvs = np.ones(int(ncomponents))

        for i in range(int(ncomponents)):

            if validvs[i] != 0.0:

                # Calculate the velocity difference between all components
                for j in range(int(ncomponents)):
                    diff[j] = abs(velolist[i]-velolist[j])
                diff[(diff==0.0)] = np.nan

                # Find the minimum difference (i.e. the adjacent component)
                idmin = np.where(diff==np.nanmin(diff))[0]
                idmin = idmin[0]
                adjacent_intensity = intlist[idmin]
                adjacent_velocity = velolist[idmin]
                adjacent_dispersion = displist[idmin]

                # Get the separation between each component and its neighbour
                sep = np.abs(velolist[i] - adjacent_velocity)
                # Calculate the allowed separation between components
                min_allowed_sep = np.min(np.array([displist[i], adjacent_dispersion]))*fwhmconv*self.tol[5]

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

        for i in range(int(ncomponents)):
            self.guesses_updated[(i*nparams)+idxp] = intlist[i]
            self.guesses_updated[(i*nparams)+idxv] = velolist[i]
            self.guesses_updated[(i*nparams)+idxd] = displist[i]

        violating_comps = (self.guesses_updated==0.0)
        if np.any(violating_comps):
            self.validfit=False
        else:
            self.validfit=True

        self.guesses_updated = self.guesses_updated[(self.guesses_updated != 0.0)]

    def find_closest_match(self,i,nparams):
        """
        Find the closest matching component in the parent SAA model to the current
        component in bf.

        Parameters
        ----------
        i : number
            index for params
        nparams : number
            number of parameters in the pyspeckit model

        """

        diff = np.zeros(int(np.size(self.guesses_parent)/nparams))
        for j in range(int(np.size(self.guesses_parent)/nparams)):
            pdiff = 0.0
            for k in range(nparams):
                pdiff+=(self.guesses_updated[int((i*nparams)+k)] - self.guesses_parent[int((j*nparams)+k)])**2.
            diff[j] = np.sqrt(pdiff)

        return diff

    def fit_converge(self):
        if None in self.pskspectrum.specfit.modelerrs:
            return False
        else:
            return True

    def printable_format(self):
        """

        Parameters
        ----------

        """
        specfit=self.pskspectrum.specfit
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
