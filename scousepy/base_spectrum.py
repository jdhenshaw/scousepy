# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np
from .stage_1 import calc_rms

class BaseSpectrum(object):
    def __init__(self, coords, flux, idx=None, scouse=None):
        """
        Stores all the information regarding individual spectra
        """

        self._index = idx
        self._coordinates = np.array(coords, dtype='int')
        self._rms = get_rms(self, scouse, flux)
        self._model_parent = None
        self._model_spatial = None
        self._model_dud = None
        self._models = None
        self._model = None
        self._flux = flux

    @property
    def flux(self):
        return self._flux

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

    def __repr__(self):
        """
        Return a nice printable format for the object.
        """
        return "<< scousepy individual spectrum; index={0} >>".format(self.index)

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
