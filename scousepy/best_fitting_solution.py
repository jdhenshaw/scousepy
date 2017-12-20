# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2017 Jonathan D. Henshaw
CONTACT: j.d.henshaw[AT]ljmu.ac.uk

"""
from astropy.stats import akaike_info_criterion as aic
import numpy as np

class fit(object):
    #TODO: Need to make this generic for pyspeckit's other models

    def __init__(self, spec, idx=None, scouse=None):
        """
        Stores the best-fitting solutions

        """
        self._fit_idx = idx
        self._x = spec.xarr
        self._y = spec.flux
        self._ncomps = spec.specfit.npeaks
        self._params = spec.specfit.modelpars
        self._errors = spec.specfit.modelerrs
        self._rms = spec.error[0]
        self._residuals = spec.specfit.residuals
        self._residstd = None
        self._chi2 = spec.specfit.chi2
        self._dof = spec.specfit.dof
        self._redchi2 = spec.specfit.chi2/spec.specfit.dof
        self._aic = get_aic(self, spec)
        self._tau = None

    @property
    def fit_idx(self):
        """
        Returns solution idx
        """
        return self._fit_idx

    @property
    def x(self):
        """
        Returns x values
        """
        return self._x

    @property
    def y(self):
        """
        Returns y values
        """
        return self._y

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
    def residuals(self):
        """
        Returns a residual array the same length as the x axis
        """
        return self._residuals

    @property
    def residstd(self):
        """
        Returns the standard deviation of the residuals
        """
        return np.std(self.residuals)

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
        return "<< scousepy best_fitting_solution; fit_index={0}; ncomps={1} >>".format(self.fit_idx, self.ncomps)

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
        print("Best-fitting solution based on previous SAA as input guess.")

        print("")
    print(self)
    print("")
    print(("Number of components: {0}").format(self.ncomps))
    print("")
    compcount=0
    for i in range(0, int(self.ncomps)):
        print(("Amplitude:  {0} +/- {1}").format(np.around(self.params[0+i*3], \
                                                 decimals=5), \
                                                 np.around(self.errors[0+i*3], \
                                                 decimals=5)))
        print(("Centroid:   {0} +/- {1}").format(np.around(self.params[1+i*3], \
                                                 decimals=5), \
                                                 np.around(self.errors[1+i*3], \
                                                 decimals=5)))
        print(("Dispersion: {0} +/- {1}").format(np.around(self.params[2+i*3], \
                                                 decimals=5), \
                                                 np.around(self.errors[2+i*3], \
                                                 decimals=5)))
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
