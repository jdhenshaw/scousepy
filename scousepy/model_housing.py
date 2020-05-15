# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np
from .stage_1 import calc_rms
import numpy as np
from astropy.stats import median_absolute_deviation

class BaseSpectrum(object):
    def __init__(self, coords, spectrum, idx=None, scouse=None, sample=False):
        """
        Stores all the information regarding individual spectra. These
        properties are shared
        """

        self._index = idx
        self._coordinates = np.array(coords, dtype='int')
        self._spectrum = spectrum
        self._rms = get_rms(self, scouse)
        self._model_parent = None
        self._model_spatial = None
        self._model_dud = None
        self._models = None
        self._model = None

    @property
    def spectrum(self):
        return self._spectrum

    @property
    def index(self):
        """
        Returns the index of the spectrum
        """
        return self._index

    @property
    def coordinates(self):
        """
        Returns the coordinates of the spectrum
        """
        return self._coordinates

    @property
    def rms(self):
        """
        Returns the spectral rms.
        """
        return self._rms

    @property
    def model_parent(self):
        """
        Returns the best-fitting model derived from the parent SAA model.
        """
        return self._model_parent

    @property
    def model_spatial(self):
        """
        Returns the best-fitting model which incorporates spatial fitting.
        """
        return self._model_spatial

    @property
    def model_dud(self):
        """
        Returns the dud
        """
        return self._model_dud

    @property
    def models(self):
        """
        Returns the list of available models
        """
        return self._models

    @property
    def model(self):
        """
        Returns the best-fitting model model to the data
        """
        return self._model

    @property
    def decision(self):
        """
        Returns the decision made in s6 for statistics
        """
        return self._decision

class saa(BaseSpectrum):
    """
    Stores all the information regarding individual spectral averaging areas
    """
    def __init__(self, coords, spectrum, idx=None, scouse=None, sample=False):
        super(saa, self).__init__(coords, spectrum, idx=idx, scouse=scouse)
        self._ytrim = trim_spectrum(self, scouse)
        self._indices = None
        self._indiv_spectra = None
        self._sample = sample
        self._cube_shape = scouse.cube.shape
        self.params=None

    @classmethod
    def from_indiv_spectrum(cls, indiv_spectrum, scouse, sample=False):
        return cls(coords=indiv_spectrum.coordinates,
                   flux=indiv_spectrum.flux,
                   scouse=scouse,
                   idx=indiv_spectrum.index,
                   sample=sample
                  )

    @property
    def ytrim(self):
        """
        Returns the spectrum.
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
        return np.ravel_multi_index(self.indices.T, self._cube_shape[1:])

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
        return "< scousepy Spectral Averaging Area SAA {0} >".format(self.index, self.coordinates)

class individual_spectrum(BaseSpectrum):

    def __init__(self, *args, **kwargs):
        super(individual_spectrum, self).__init__(*args, **kwargs)

        self._decision = 'original'

    @property
    def decision(self):
        """
        Returns the decision made in s6 for statistics
        """
        return self._decision

    def __repr__(self):
        """
        Return a nice printable format for the object.
        """
        return "<< scousepy individual spectrum; index={0} >>".format(self.index)

def get_rms(self, scouse):
    """
    Calculates rms value
    """
    if not np.isnan(self.spectrum).any() and not (self.spectrum > 0).all():
        rms = calc_rms(self.spectrum)
    else:
        rms = scouse.rms_approx

    return rms

def trim_spectrum(self, scouse):
    """
    Trims a spectrum according to the user inputs
    """
    return self.spectrum[scouse.trimids]

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

def add_model_parent(self, model):
    """
    Adds best-fitting model information to the spectrum - note this only adds
    the model derived from the SAA
    """
    self._model_parent = model

def add_model_spatial(self, model):
    """
    Adds best-fitting model information to the spectrum - note this only adds
    the model derived from the SAA
    """
    self._model_spatial = model

def add_model_dud(self, model):
    """
    We want to add a dud to every spectrum
    """
    self._model_dud = model

def add_bf_model(self, model):
    """
    Selected best fit model solution
    """
    self._model = model

def update_model_list(self, models):
    """
    Compiles all models
    """
    self._model_parent = None
    self._model_dud = None
    self._model_spatial = None
    self._models = models

def update_model_list_remdup(self, models):
    """
    updates model list following removal of duplicates
    """
    self._models = models

def add_decision(self, decision):
    """
    Updates the spectrum with decision made in s6
    """
    self._decision = decision
