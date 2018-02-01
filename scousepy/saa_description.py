# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np

class saa(object):
    def __init__(self, coords, flux, \
                 idx=None, scouse=None, sample = False):
        """
        Stores all the information regarding individual spectral averaging areas

        """

        self._index = idx
        self._coordinates = np.array(coords)
        self._x = np.array(scouse.cube.world[:,0,0][0])
        self._y = flux
        self._xtrim, self._ytrim = trim_spectrum(self, scouse)
        self._rms = None
        self._indices = None
        self._indices_flat = None
        self._model = None
        self._indiv_spectra = None
        self._sample = sample

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
    def model(self):
        """
        Returns the best-fitting model to the spectral averaging area.
        """
        return self._model

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
        return "<< scousepy SAA; index={0} >>".format(self.index)

def trim_spectrum(self, scouse=None):
    """
    Trims a spectrum according to the user inputs
    """
    keep = ((self.x>scouse.ppv_vol[0])&(self.x<scouse.ppv_vol[1]))
    xtrim = self.x[keep]
    ytrim = self.y[keep]
    return xtrim, ytrim

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
        idx_x, idx_y = int(self.indices[k,1]),int(self.indices[k,0])
        idx_flat = int(idx_x*scouse.cube.shape[1]+idx_y)
        indices_flat.append(idx_flat)

    self._indices_flat = np.asarray(indices_flat)

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
