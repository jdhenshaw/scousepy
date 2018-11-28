# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np
import pyspeckit
import sys
import warnings
from astropy import log
from astropy.stats import akaike_info_criterion as aic

class fit(object):
    #TODO: Need to make this generic for pyspeckit's other models

    def __init__(self, spec, idx=None, scouse=None, fit_dud=False, noise=None,
                 duddata=None):
        """
        Stores the best-fitting model

        """

        self._index = idx

        if fit_dud:
            spec=None
            #quickly and quietly generate a dud spec
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                old_log = log.level
                log.setLevel('ERROR')

                spec = pyspeckit.Spectrum(data=[0,0], error=[0,0], xarr=[0,0])
                spec.specfit.fittype = scouse.fittype
                spec.specfit.fitter = spec.specfit.Registry.multifitters[scouse.fittype]

                log.setLevel(old_log)

            fit_pars_dud(self, spec, scouse, noise, duddata)
        else:
            fit_pars(self, spec, scouse)


    @property
    def index(self):
        """
        Returns model idx
        """
        return self._index

    @property
    def fittype(self):
        """
        Returns pyspeckit fittype - i.e. the type of model
        """
        return self._fittype

    @property
    def parnames(self):
        """
        Returns pyspeckit parnames
        """
        return self._parnames

    @property
    def ncomps(self):
        """
        Returns the number of fitted components
        """
        return self._ncomps

    @property
    def params(self):
        """
        Returns model parameters
        """
        return self._params

    @property
    def errors(self):
        """
        Returns errors on model params
        """
        return self._errors

    @property
    def rms(self):
        """
        Returns the rms noise of the spectrum
        """
        return self._rms

    @property
    def residstd(self):
        """
        Returns the standard deviation of the residuals
        """
        return self._residstd

    @property
    def chi2(self):
        """
        Returns chisq
        """
        return self._chi2

    @property
    def dof(self):
        """
        Returns number of degrees of freedom
        """
        return self._dof

    @property
    def redchi2(self):
        """
        Returns reduced chisq
        """

        return self._redchi2

    @property
    def aic(self):
        """
        Returns AIC value
        """
        return self._aic

    def __repr__(self):
        """
        Return a nice printable format for the object.

        """
        return "<< scousepy model_solution; index={0}; ncomps={1} >>".format(self.index, self.ncomps)

def fit_pars_dud(self, spec, scouse, noise, duddata):
    """
    Sets the parameters for a dud spectrum
    """

    self._fittype = spec.specfit.fittype
    self._parnames = spec.specfit.fitter.parnames
    self._ncomps = 0.0
    self._params = [0.0 for i in range(len(self.parnames))]
    self._errors = [0.0 for i in range(len(self.parnames))]
    self._rms = noise
    self._residstd = np.std(duddata)
    self._dof = 0.0
    self._chi2 = 0.0
    self._redchi2 = 0.0
    self._aic = 0.0

def fit_pars(self, spec, scouse):
    """
    Sets the parameters for a spectrum
    """
    self._fittype = spec.specfit.fittype
    self._parnames = spec.specfit.fitter.parnames
    self._ncomps = spec.specfit.npeaks
    self._params = spec.specfit.modelpars
    self._errors = spec.specfit.modelerrs
    self._rms = spec.error[0]
    self._residstd = np.std(spec.specfit.residuals)
    self._chi2 = spec.specfit.chi2
    self._dof = spec.specfit.dof
    self._redchi2 = spec.specfit.chi2/spec.specfit.dof
    self._aic = get_aic(self, spec)

def get_aic(self, spec):
    """
    Calculates the AIC value from the spectrum
    """
    logl = spec.specfit.fitter.logp(spec.xarr, spec.data, spec.error)
    return aic(logl, self.ncomps+(self.ncomps*3.), len(spec.xarr))

def print_fit_information(self, init_guess=False):
    """
    Prints fit information to terminal
    """
    print("=============================================================")
    if init_guess:
        print("Best-fitting model based on previous SAA as input guess.")
        print("")

    print(self)
    print("")
    print("Model type: {0}".format(self.fittype))
    print("")
    print(("Number of components: {0}").format(self.ncomps))
    print("")
    compcount=0
    for i in range(0, int(self.ncomps)):
        parlow = int((i*len(self.parnames)))
        parhigh = int((i*len(self.parnames))+len(self.parnames))
        parrange = np.arange(parlow,parhigh)
        for j in range(0, len(self.parnames)):
            # print(("{0}:  {1} +/- {2}").format(self.parnames[j], \
            #                                    np.around(self.params[parrange[j]],
            #                                    decimals=5), \
            #                                    np.around(self.errors[parrange[j]],
            #                                    decimals=5)))

            print(("{0}:  {1} +/- {2}").format(self.parnames[j], \
                                               self.params[parrange[j]],
                                               self.errors[parrange[j]]))
        print("")
        compcount+=1
    print(("chisq:    {0}").format(np.around(self.chi2, decimals=2)))
    print(("redchisq: {0}").format(np.around(self.redchi2, decimals=2)))
    print(("AIC:      {0}").format(np.around(self.aic, decimals=2)))
    print("")

    if init_guess:
        print("To enter interative fitting mode type 'f'")
        print("")

    print("=============================================================")
