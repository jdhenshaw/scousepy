# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np

def model_selection(scouseobject):
    """
    Selects the best model out of those fitted - that with the smallest aic
    value

    Parameters
    ----------
    scouseobject : Instance of the scousepy class

    """
    from .verbose_output import print_to_terminal
    from .model_housing2 import indivmodel

    if scouseobject.verbose:
        progress_bar = print_to_terminal(stage='s4', step='selectmodsstart',length=len(scouseobject.indiv_dict.keys()))

    for key in scouseobject.indiv_dict.keys():
        indivspec = scouseobject.indiv_dict[key]
        models = indivspec.model_from_parent
        findduds = [True if model is None else False for model in models]
        if np.any(np.asarray(findduds)):
            idx = np.squeeze(np.where(np.asarray(findduds) == True))
            if np.size(idx)>1:
                for j in range(len(idx)):
                    dud = models[idx[j]]
                    models.remove(dud)
            else:
                dud = models[idx]
                models.remove(dud)

        if np.size(models) != 0:
            aic = [model.AIC for model in models]
            idx = np.squeeze(np.where(aic == np.min(aic)))
            bfmodel = models[idx]
            models.remove(bfmodel)
        else:
            modeldict = create_a_dud(indivspec)
            bfmodel = indivmodel(modeldict)

        setattr(indivspec, 'model', bfmodel)

        if scouseobject.verbose:
            progress_bar.update()

def create_a_dud(indivspec):
    """
    Creates a dud spectrum - used if no best fitting solution can be found
    """
    modeldict={}
    modeldict['fittype']=None
    modeldict['parnames']=['amplitude','shift','width']
    modeldict['ncomps']=0
    modeldict['params']=[0.0,0.0,0.0]
    modeldict['errors']=[0.0,0.0,0.0]
    modeldict['rms']=indivspec.rms
    modeldict['residstd']= np.std(indivspec.spectrum)
    modeldict['chisq']=0.0
    modeldict['dof']=0.0
    modeldict['redchisq']=0.0
    modeldict['AIC']=0.0
    modeldict['fitconverge']=False
    modeldict['method']='dud'

    return modeldict
