# Licensed under an MIT open source license - see LICENSE

import numpy as np
import sys
from astropy.table import Table
from astropy.table import Column

from .io import get_headings

class stats(object):
    def __init__(self, scouseobject=None):
        """
        Computes basic statistics on fitting
        """
        self._nspec = get_nspec(self, scouseobject)
        self._nsaa, self._nsaa_indiv = get_nsaa(self,scouseobject)
        self._nspecsaa, self._nspecsaa_indiv = get_nspecsaa(self, scouseobject)
        self._nfits = get_nfits(self, scouseobject)
        self._ncomps = get_ncomps(self, scouseobject)
        self._ncomps_saa = get_ncomps_saa(self, scouseobject)
        self._ncompsperfit = None
        self._ncompsperfit_saa = None
        self._noriginal = get_noriginal(self, scouseobject)
        self._nrefit = get_nrefit(self, scouseobject)
        self._nalt = get_nalt(self, scouseobject)
        self._nmultiple = get_nmultiple(self, scouseobject)
        self._nmultiple_saa = get_nmultiple_saa(self, scouseobject)
        self._originalfrac = None
        self._refitfrac = None
        self._altfrac = None
        self._saastats = get_saa_stats(self, scouseobject)
        self._stats = get_param_stats(self, scouseobject)
        self._meanrms = None
        self._meanresid = None
        self._residratio = get_residratio(self, scouseobject)
        self._meanchisq = None
        self._meanredchisq = None
        self._meanAIC = None

    @property
    def stats(self):
        """
        Returns a dictionary containing the statistics of a fitted parameter
        """
        return self._stats

    @property
    def saastats(self):
        """
        Returns a dictionary containing the statistics of saa fitting
        """
        return self._saastats

    @property
    def meanAIC(self):
        """
        Returns mean chiq
        """
        return self._stats['AIC'][5]

    @property
    def meanredchisq(self):
        """
        Returns mean redchisq
        """
        return self._stats['redchisq'][5]

    @property
    def meanchisq(self):
        """
        Returns mean chisq
        """
        return self._stats['chisq'][5]

    @property
    def residratio(self):
        """
        Returns mean resid
        """
        return self._residratio

    @property
    def meanresid(self):
        """
        Returns mean resid
        """
        return self._stats['residual'][5]

    @property
    def meanrms(self):
        """
        Returns mean rms
        """
        return self._stats['rms'][5]

    @property
    def nmultiple(self):
        """
        Number of spectra requiring multicomponent fits
        """
        return self._nmultiple

    @property
    def nmultiple_saa(self):
        """
        Number of spectra requiring multicomponent fits
        """
        return self._nmultiple_saa

    @property
    def noriginal(self):
        """
        Number of spectra fitted during the original run
        """
        return self._noriginal

    @property
    def nrefit(self):
        """
        Number of spectra refitted
        """
        return self._nrefit

    @property
    def nalt(self):
        """
        Number of spectra where alternative solutions were selected
        """
        return self._nalt

    @property
    def originalfrac(self):
        """
        Fraction of spectra fitted during the original run
        """
        return self._noriginal/float(self.noriginal+self.nrefit+self.nalt)

    @property
    def refitfrac(self):
        """
        Fraction of spectra refitted
        """
        return self._nrefit/float(self.noriginal+self.nrefit+self.nalt)

    @property
    def altfrac(self):
        """
        Fraction of spectra where alternative solutions were selected
        """
        return self._nalt/float(self.noriginal+self.nrefit+self.nalt)


    @property
    def ncompsperfit(self):
        """
        Returns number of components per fitted position
        """
        return self._ncomps/float(self._nfits)

    @property
    def ncompsperfit_saa(self):
        """
        Returns number of components per fitted position
        """

        return np.asarray(self._ncomps_saa)/np.asarray(self._nsaa_indiv)

    @property
    def ncomps(self):
        """
        Number of components fitted
        """
        return self._ncomps

    @property
    def ncomps_saa(self):
        """
        Number of components fitted
        """
        return self._ncomps_saa

    @property
    def nfits(self):
        """
        Number of non-duds
        """
        return self._nfits

    @property
    def nspecsaa(self):
        """
        Number of spectra in saas
        """
        return self._nspecsaa

    @property
    def nspecsaa_indiv(self):
        """
        Number of spectra in saas
        """
        return self._nspecsaa_indiv

    @property
    def nsaa(self):
        """
        Returns number of spectral averaging areas
        """
        return self._nsaa

    @property
    def nsaa_indiv(self):
        """
        Returns number of spectral averaging areas
        """
        return self._nsaa_indiv

    @property
    def nspec(self):
        """
        Total number of spectra in cube
        """
        return self._nspec

def get_saa_stats(self, scouseobject):
    """
    Calculates statistics for saa Fitting
    """
    keys = list(scouseobject.saa_dict.keys())

    statkeys = ['SNR', 'alpha']
    stat_dict = {}
    for key in keys:
        saa_dict=scouseobject.saa_dict[key]

        for i, saa in saa_dict.items():
            if saa.to_be_fit:
                nparams = np.size(saa.model.parnames)
                break

        ncomps = []
        params = []
        errors = []
        rms = []
        residstd = []
        chisq = []
        redchisq = []
        AIC = []
        SNR = []
        alpha = []
        method = []

        for i, saa in saa_dict.items():
            if saa.to_be_fit:
                ncomps.append(saa.model.ncomps)
                rms.append(saa.rms)
                residstd.append(saa.model.residstd)
                chisq.append(saa.model.chisq)
                redchisq.append(saa.model.redchisq)
                AIC.append(saa.model.AIC)

                for i in range(0, int(saa.model.ncomps)):
                    params.append(saa.model.params[(i*len(saa.model.parnames)):(i*len(saa.model.parnames))+len(saa.model.parnames)])
                    errors.append(saa.model.errors[(i*len(saa.model.parnames)):(i*len(saa.model.parnames))+len(saa.model.parnames)])

                if saa.model.method!='manual':
                    SNR.append(saa.model.SNR)
                    alpha.append(saa.model.alpha)
                else:
                    method.append(saa.model.method)

        commonstats = [ncomps, rms, residstd, chisq, redchisq, AIC, SNR, alpha]
        commonstatkeys = ['ncomps','rms','residstd','chisq','redchisq','AIC','SNR','alpha']
        statkeys = get_statkeys(scouseobject, nparams, saa=True)
        statlist = unpack_statlist(nparams, commonstats, commonstatkeys, params, errors, statkeys)

        statdict = get_stat_dict(statkeys, statlist)
        statdict['nmanual']=float(method.count('manual'))

        stat_dict[key]=statdict

    return stat_dict


def get_param_stats(self, scouseobject):
    """
    Calculates statistics for parameters
    """

    keys = list(scouseobject.indiv_dict.keys())
    nparams = np.size(scouseobject.indiv_dict[keys[0]].model.parnames)

    ncomps = []
    params = []
    errors = []
    rms = []
    residstd = []
    chisq = []
    redchisq = []
    AIC = []

    for key in keys:
        spectrum = scouseobject.indiv_dict[key]
        if spectrum.model is not None:
            if spectrum.model.ncomps != 0.0:
                ncomps.append(spectrum.model.ncomps)
                rms.append(spectrum.rms)
                residstd.append(spectrum.model.residstd)
                chisq.append(spectrum.model.chisq)
                redchisq.append(spectrum.model.redchisq)
                AIC.append(spectrum.model.AIC)

                for i in range(0, int(spectrum.model.ncomps)):
                    params.append(spectrum.model.params[(i*len(spectrum.model.parnames)):(i*len(spectrum.model.parnames))+len(spectrum.model.parnames)])
                    errors.append(spectrum.model.errors[(i*len(spectrum.model.parnames)):(i*len(spectrum.model.parnames))+len(spectrum.model.parnames)])

    commonstats = [ncomps, rms, residstd, chisq, redchisq, AIC]
    commonstatkeys = ['ncomps','rms','residstd','chisq','redchisq','AIC']
    statkeys = get_statkeys(scouseobject, nparams)
    statlist = unpack_statlist(nparams, commonstats, commonstatkeys, params, errors, statkeys)

    stat_dict = get_stat_dict(statkeys, statlist)

    return stat_dict

def get_statkeys(scouseobject, nparams, saa=False):
    """
    Returns key headings for the stats
    """
    statkeys = get_headings(scouseobject, scouseobject.saa_dict[0])
    statkeys.remove('x')
    statkeys.remove('y')
    statkeys.remove('dof')
    _ncomps = statkeys.pop(0)
    statkeys.insert((nparams*2), _ncomps)
    if saa:
        statkeys.append('SNR')
        statkeys.append('alpha')

    return statkeys

def unpack_statlist(nparams, commonstats, commonstatkeys, params, errors, statkeys):
    """
    Creates a list of stats in the same order as statkeys
    """
    _statlist = []
    params = np.asarray(params,dtype='object')
    errors = np.asarray(errors,dtype='object')
    for i in range(nparams):
        _statlist.append(params[:,i])
        _statlist.append(errors[:,i])

    for i in range(np.size(commonstatkeys)):
        stat=np.asarray(commonstats[i])
        _statlist.append(stat)

    _statlist = np.asarray(_statlist,dtype='object')

    return _statlist

def get_stat_dict(statkeys, statlist):
    """
    Returns min, first quartile, median, third quartile, max, mean for a given
    statistic
    """
    stat_dict = {}

    for i in range(np.size(statkeys)):
        key = statkeys[i]
        stat = np.asarray(statlist[i])
        stat = np.asarray(statlist[i], dtype=np.float64)
        stat_dict[key] = [np.nanmin(stat), \
                          np.nanpercentile(stat, 25), \
                          np.nanmedian(stat),\
                          np.nanpercentile(stat, 75),\
                          np.nanmax(stat),\
                          np.nanmean(stat)]

    return stat_dict

def get_residratio(self, scouseobject):
    """
    Calculates mean ratio of resid/rms
    """
    rmslist = [scouseobject.indiv_dict[key].model.rms for key in scouseobject.indiv_dict.keys() if (scouseobject.indiv_dict[key].model is not None) and (scouseobject.indiv_dict[key].model.ncomps != 0.0) ]
    residlist = [scouseobject.indiv_dict[key].model.residstd for key in scouseobject.indiv_dict.keys() if (scouseobject.indiv_dict[key].model is not None) and (scouseobject.indiv_dict[key].model.ncomps != 0.0) ]
    return (np.asarray(residlist)/np.asarray(rmslist)).mean(axis=0)

def get_nfits(self, scouseobject):
    """
    Calculates number of non-dud fits
    """
    fits = [scouseobject.indiv_dict[key].model.ncomps for key in scouseobject.indiv_dict.keys() if (scouseobject.indiv_dict[key].model is not None) and (scouseobject.indiv_dict[key].model.ncomps != 0.0)]
    return np.size(fits)

def get_nspec(self, scouseobject):
    """
    Gets number of spectra
    """
    return scouseobject.cube.shape[1]*scouseobject.cube.shape[2]

def get_ncomps(self, scouseobject):
    """
    Calculates number of components
    """
    fits = [scouseobject.indiv_dict[key].model.ncomps for key in scouseobject.indiv_dict.keys() if (scouseobject.indiv_dict[key].model is not None) and (scouseobject.indiv_dict[key].model.ncomps != 0.0) ]
    return np.sum(fits)

def get_nmultiple(self, scouseobject):
    """
    Calculates number of components
    """
    fits = [scouseobject.indiv_dict[key].model.ncomps for key in scouseobject.indiv_dict.keys() if (scouseobject.indiv_dict[key].model is not None) and (scouseobject.indiv_dict[key].model.ncomps > 1.0)  ]
    return len(fits)

def get_ncomps_saa(self, scouseobject):
    """
    Calculates number of components
    """
    fits=[]
    keys = list(scouseobject.saa_dict.keys())
    for key in keys:
        saa_dict=scouseobject.saa_dict[key]
        _fits = []
        for i, saa in saa_dict.items():
            if saa.to_be_fit:
                if saa.model.ncomps!=0:
                    _fits.append(saa.model.ncomps)
        fits.append(np.sum(_fits))
    return fits

def get_nmultiple_saa(self, scouseobject):
    """
    Calculates number of components
    """
    fits=[]
    keys = list(scouseobject.saa_dict.keys())
    for key in keys:
        saa_dict=scouseobject.saa_dict[key]
        _fits = []
        for i, saa in saa_dict.items():
            if saa.to_be_fit:
                if saa.model.ncomps>1:
                    _fits.append(saa.model.ncomps)
        fits.append(len(_fits))
    return fits

def get_nspecsaa(self, scouseobject):
    """
    Number of spectra contained within spectral averaging areas
    """
    indices = []
    indices_indiv_wsaa = []
    nspecsaa_indiv = []

    for i, r in enumerate(scouseobject.wsaa, start=0):
        saa_dict = scouseobject.saa_dict[i]
        for key in saa_dict.keys():
            SAA = saa_dict[key]
            if SAA.to_be_fit:
                indices+=list(SAA.indices_flat)
                indices_indiv_wsaa+=list(SAA.indices_flat)
        indices = np.asarray(indices)
        indices = np.unique(indices)
        indices = list(indices)

        indices_indiv_wsaa = np.asarray(indices_indiv_wsaa)
        indices_indiv_wsaa = np.unique(indices_indiv_wsaa)
        indices_indiv_wsaa = list(indices_indiv_wsaa)

        nspecsaa_indiv.append(np.size(indices_indiv_wsaa))
        indices_indiv_wsaa = []

    # Total number of spectra contained within the spectral averaging areas
    nspecsaa = np.size(indices)

    return nspecsaa, nspecsaa_indiv

def get_nsaa(self, scouseobject):
    """
    Total number of spectral averaging areas that have been fitted
    """
    nsaa = 0.0
    nsaa_indiv = []
    for i, r in enumerate(scouseobject.wsaa, start=0):
        saa_dict = scouseobject.saa_dict[i]
        for key in saa_dict.keys():
            SAA = saa_dict[key]
            if SAA.to_be_fit:
                nsaa+=1
        nsaa_indiv.append(nsaa)
        nsaa = 0.0
    return np.sum(nsaa_indiv), nsaa_indiv

def get_nrefit(self, scouseobject):
    """
    Get the number of spectra that were manually refitted.
    """

    nrefit = 0.0
    for key in scouseobject.indiv_dict.keys():
        spectrum = scouseobject.indiv_dict[key]
        if spectrum.decision == 'refit':
            nrefit+=1

    return nrefit

def get_nalt(self, scouseobject):
    """
    Get the number of spectra that were manually refitted.
    """

    nalt = 0.0
    for key in scouseobject.indiv_dict.keys():
        spectrum = scouseobject.indiv_dict[key]
        if spectrum.decision == 'alternative':
            nalt+=1

    return nalt

def get_noriginal(self, scouseobject):
    """
    Get the number of spectra that were manually refitted.
    """

    noriginal = 0.0
    for key in scouseobject.indiv_dict.keys():
        spectrum = scouseobject.indiv_dict[key]
        if spectrum.decision == 'original':
            noriginal+=1

    return noriginal
