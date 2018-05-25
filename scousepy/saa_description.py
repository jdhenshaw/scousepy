# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np
from astropy.stats import median_absolute_deviation

from .stage_1 import calc_rms

class saa(object):

    def __init__(self, scouse, flux, sample=False, **kwargs):
        super(self, BaseSpectrum).__init__(scouse, flux, **kwargs)
        """
        Stores all the information regarding individual spectral averaging areas

        """
        self._ytrim = trim_spectrum(self, scouse, flux)
        self._indices = None
        self._indices_flat = None
        self._indiv_spectra = None
        self._sample = sample

    @property
    def ytrim(self):
        """
        Returns the spectrum of the spectral averaging area.
        """
        return self._ytrim

    @property
    def indices(self):
        """
        Returns the individual indices contained within the spectral
        averaging area.
        """
        return self._indices

    @property
    def indices_flat(self):
        """
        Returns the flattened individual indices contained within the spectral
        averaging area.
        """
        return self._indices_flat

    @property
    def to_be_fit(self):
        """
        Indicates whether or not the spectrum is to be fit (used for training
        set generation)
        """
        return self._sample

    @property
    def indiv_spectra(self):
        """
        Returns a dictionary containing the models to the individual spectra
        contained within the SAA
        """
        return self._indiv_spectra

    def __repr__(self):
        """
        Return a nice printable format for the object.
        """
        return "< SAA {0} >".format(self.index, self.coordinates)

def get_rms(self, scouse, flux):
    """
    Calculates rms value
    """
    spectrum = flux
    if not np.isnan(spectrum).any() and not (spectrum > 0).all():
        rms = calc_rms(spectrum)
    else:
        rms = scouse.rms_approx

    return rms

def trim_spectrum(self, scouse, flux):
    """
    Trims a spectrum according to the user inputs
    """
    return flux[scouse.trimids]

def add_model(self, model):
    """
    Adds best-fitting model information to the SAA
    """
    self._model = model

def add_ids(self, ids):
    """
    Adds indices contained within the SAA
    """
    self._indices = np.array(ids, dtype='int')

def add_flat_ids(self, scouse=None):
    """
    Flattens indices
    """
    indices_flat = []
    for k in range(len(self.indices[:,0])):
        idx_y, idx_x = int(self.indices[k,0]),int(self.indices[k,1])
        idx_flat = int(idx_x*scouse.cube.shape[1]+idx_y)
        indices_flat.append(idx_flat)

    self._indices_flat = np.asarray(indices_flat)
    self._indices = None

def add_indiv_spectra(self, dict):
    """
    Adds indices contained within the SAA
    """
    self._indiv_spectra = dict

def merge_models(self, merge_spec):
    """
    Merges merge_spec models into self
    """
    main_models = self.models
    merge_models = merge_spec.models
    allmodels = []
    allmodels.append(main_models)
    allmodels.append(merge_models)
    self._models = [model for mergemods in allmodels for model in mergemods]

def clean_up(self):
    """
    Cleans model solutions
    """
    self._indiv_spectra = None
