# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np
from .saa_description import get_rms
from .base_spectrum import BaseSpectrum

class spectrum(BaseSpectrum):
    
    def __init__(self, *args, **kwargs):
        super(spectrum, self).__init__(*args, **kwargs)

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
