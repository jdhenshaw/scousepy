# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np

class spectrum(object):
    def __init__(self, coords, flux, \
                 idx=None, scouse=None):
        """
        Stores all the information regarding individual spectral averaging areas

        """

        self._index = idx
        self._coordinates = np.array(coords)
        self._x = np.array(scouse.cube.world[:,0,0][0])
        self._y = flux
        self._xtrim, self._ytrim = trim_spectrum(self, scouse)
        self._rms = None
        self._model_parent = None
        self._model_spatial = None
        self._model_dud = None
        self._models = None
        self._model = None

    @property
    def index(self):
        """
        Returns the index of the spectral averaging area.
        """
        return self._index

    @property
    def coordinates(self):
        """
        Returns the coordinates of the spectral averaging area.
        """
        return self._coordinates

    @property
    def x(self):
        """
        Returns the spectrum of the spectral averaging area.
        """
        return self._x

    @property
    def y(self):
        """
        Returns the spectrum of the spectral averaging area.
        """
        return self._y

    @property
    def xtrim(self):
        """
        Returns the spectrum of the spectral averaging area.
        """
        return self._xtrim

    @property
    def ytrim(self):
        """
        Returns the spectrum of the spectral averaging area.
        """
        return self._ytrim

    @property
    def rms(self):
        """
        Returns the spectral rms.
        """
        # Find all negative values
        yneg = self.y[(self.y < 0.0)]
        # Get the mean/std
        mean = np.mean(yneg)
        std = np.std(yneg)
        # maximum neg = 4 std from mean
        maxneg = mean-4.*std
        # compute std over all values within that 4sigma limit and return
        return np.std(self.y[self.y < abs(maxneg)])

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

    def __repr__(self):
        """
        Return a nice printable format for the object.
        """
        return "<< scousepy individual spectrum; index={0} >>".format(self.index)

def trim_spectrum(self, scouse=None):
    """
    Trims a spectrum according to the user inputs
    """
    keep = ((self.x>scouse.ppv_vol[0])&(self.x<scouse.ppv_vol[1]))
    xtrim = self.x[keep]
    ytrim = self.y[keep]
    return xtrim, ytrim

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
