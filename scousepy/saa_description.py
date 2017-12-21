# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2017 Jonathan D. Henshaw
CONTACT: j.d.henshaw[AT]ljmu.ac.uk

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
        self._xtrim = self.x
        self._ytrim = self.y
        self._rms = None
        self._indices = None
        self._solution = None
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
        return self._rms

    @property
    def indices(self):
        """
        Returns the individual indices contained within the spectral
        averaging area.
        """
        return self._indices

    @property
    def solution(self):
        """
        Returns the best-fitting solution to the spectral averaging area.
        """
        return self._solution

    @property
    def to_be_fit(self):
        """
        Indicates whether or not the spectrum is to be fit (used for training
        set generation)
        """
        return self._sample

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
    self._xtrim = self.x[keep]
    self._ytrim = self.y[keep]

def get_noise(self):
    """
    Works out rms noise within a spectrum
    """
    # Find all negative values
    yneg = self.y[(self.y < 0.0)]
    # Get the mean/std
    mean = np.mean(yneg)
    std = np.std(yneg)
    # maximum neg = 4 std from mean
    maxneg = mean-4.*std
    # compute std over all values within that 4sigma limit
    self._rms = np.std(self.y[self.y < abs(maxneg)])

def add_solution(self, solution):
    """
    Adds best-fitting solution information to the SAA
    """
    self._solution = solution

def add_ids(self, ids):
    """
    Adds indices contained within the SAA
    """
    self._indices = np.array(ids)
