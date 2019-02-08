# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np
from .indiv_spec_description import *

def select_best_model(scouseobject):
    """
    Selects the best model out of those fitted - that with the smallest aic
    value

    Parameters
    ----------
    scouseobject : Instance of the scousepy class
    
    """

    for key in scouseobject.indiv_dict.keys():
        spectrum = scouseobject.indiv_dict[key]
        models = spectrum.models
        ncomps = [mod.ncomps for mod in models]
        findduds = (np.asarray(ncomps) == 0.0)

        if np.any(np.asarray(findduds)):
            idx = np.squeeze(np.where(findduds == True))
            dud = models[idx]
            models.remove(dud)
        else:
            dud=None
        if np.size(models) != 0:
            aic = [mod.aic for mod in models]
            idx = np.squeeze(np.where(aic == np.min(aic)))
            model = models[idx]
            models.remove(model)
        else:
            model = dud
            dud = None

        if dud is None:
            add_bf_model(scouseobject.indiv_dict[key], model)
            update_model_list(scouseobject.indiv_dict[key], models)
        else:
            models.append(dud)
            add_bf_model(scouseobject.indiv_dict[key], model)
            update_model_list(scouseobject.indiv_dict[key], models)
