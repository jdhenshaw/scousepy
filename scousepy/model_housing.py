# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""
import numpy as np

class saa(object):
    """
    Stores all the information regarding individual spectral averaging areas
    (SAA)

    Parameters
    ----------
    coordinates : array
        The coordinates of the SAA in pixel units. In (x,y).
    spectrum : array
        The spectrum
    index : number
        The index of the spectral averaging area (used as a key for saa_dict)
    to_be_fit : bool
        Indicating whether or not the SAA is to be fit or not
    scouseobject : instance of the scouse class

    Attributes
    ----------
    index : number
        The index of the SAA
    coordinates : array
        The coordinates of the SAA in pixel units
    spectrum : array
        The spectrum
    rms : number
        An estimate of the rms
    indices : array
        The indices of individual pixels located within the SAA
    indices_flat : array
        The same indices but flattened according to the shape of the cube
    to_be_fit : bool
        Indicating whether or not the SAA is to be fit or not
    individual_spectra : dictionary
        This is a dictionary which will contain a number of instances of the
        "individual_spectrum" class. individual spectra are initially housed
        in the saa object. After the fitting has completed these will be
        moved into a single dictionary and this dictionary will be removed

    """
    def __init__(self, coordinates, spectrum, index=None, to_be_fit=False, scouseobject=None):

        self.index=index
        self.coordinates=coordinates
        self.spectrum=spectrum
        self.rms=get_rms(self,scouseobject)
        self.indices=None
        self.indices_flat=None
        self.to_be_fit=to_be_fit
        self.model=None
        self.individual_spectra=None

    def __repr__(self):
        """
        Return a nice printable format for the object.
        """
        return "< scousepy SAA; index={0} >".format(self.index)

    def add_indices(self, indices, shape):
        """
        Adds indices contained within the SAA

        Parameters
        ----------
        indices : ndarray
            An array containing the indices that are to be added to the SAA
        shape : ndarray
            X, Y shape of the data cube. Used to flatten the indices

        """
        self.indices=np.array(indices, dtype='int')
        self.indices_flat=np.ravel_multi_index(indices.T, shape)

    def add_saamodel(self, model):
        """
        Adds model solution as described by saa model to the SAA

        Parameters
        ----------
        model : instance of the saamodel class

        """
        self.model=model


class individual_spectrum(object):
    """
    Stores all the information regarding individual spectra

    Parameters
    ----------
    coordinates : array
        The coordinates of the spectrum in pixel units. In (x,y).
    spectrum : array
        The spectrum
    index : number
        The flattened index of the spectrum
    scouseobject : instance of the scouse class
    saa_dict_index : number
        Index of the saa_dict. This will be used along with saaindex to find the
        parent model solution to provide initial guesses for the fitter
    saaindex : number
        Index of the SAA. Used to locate a given spectrum's parent SAA

    Attributes
    ----------
    template : instance of pyspeckit's Spectrum class
        A template spectrum updated during fitting
    model : instance of the indivmodel class
        The final best-fitting model solution as determined in stage 4
    model_from_parent : instance of the indivmodel class
        The best-fitting solution as determined from using the SAA model as
        input guesses
    model_from_dspec : instance of the indivmodel class
        The best-fitting model solution derived from derivative spectroscopy
    model_from_spatial : instance of the indivmodel class
        The best-fitting model solution derived from spatial fitting
    model_from_manual : instance of the indivmodel class
        The best-fitting model solution as fit manually during stage 6 of the
        process
    decision : string
        The decision made during stage 6 of the process, i.e. if the spectrum
        was refit,
    """

    def __init__(self, coordinates, spectrum, index=None, scouseobject=None,
                 saa_dict_index=None, saaindex=None):

        self.index=index
        self.coordinates=coordinates
        self.spectrum=spectrum
        self.rms=get_rms(self, scouseobject)
        self.saa_dict_index=saa_dict_index
        self.saaindex=saaindex
        self.template=None
        self.guesses_from_parent=None
        self.guesses_updated=None
        self.model=None
        self.model_from_parent=None
        self.model_from_dspec=None
        self.model_from_spatial=None
        self.model_from_manual=None
        self.decision=None

    def __repr__(self):
        """
        Return a nice printable format for the object.
        """
        return "<< scousepy individual spectrum; index={0} >>".format(self.index)

    def add_model(self, model):
        """
        Adds model solution

        Parameters
        ----------
        model : instance of the indivmodel class

        """

        if model.method=='parent':
            self.model_from_parent=model
        elif model.method=='dspec':
            self.model_from_dspec=model
        elif model.method=='spatial':
            self.model_from_spatial=model
        elif model.method=='manual':
            self.model_from_manual=model
        else:
            pass # error here?

def get_rms(self, scouseobject):
    """
    Calculates rms value. Used by both saa and individual_spectrum classes

    Parameters
    ----------
    scouseobject : instance of the scouse class

    """
    from scousepy.noisy import getnoise
    noisy=getnoise(scouseobject.x, self.spectrum)

    if np.isfinite(noisy.rms):
        rms = noisy.rms
    else:
        # if the spectrum violates these conditions then simply set the rms to
        # the value measured over the entire cube
        rms = scouseobject.rms_approx

    return rms

class basemodel(object):
    """
    Base model for scouse. These properties are shared by both SAA model
    solutions and individual spectra solutions

    Attributes
    ----------
    fittype : string
        Model used during fitting (e.g. Gaussian)
    parnames : list
        A list containing the parameter names in the model (corresponds to those
        used in pyspeckit)
    ncomps : Number
        Number of components in the model solution
    params : list
        The parameter estimates
    errors : list
        The uncertainties on each measured parameter
    rms : Number
        The measured rms value
    residstd : Number
        The standard deviation of the residuals
    chisq : Number
        The chi squared value
    dof : Number
        The number of degrees of freedom
    redchisq : Number
        The reduced chi squared value
    AIC : Number
        The akaike information criterion
    fitconverge : bool
        Indicates whether or not the fit has converged

    """

    def __init__(self):

        self.fittype=None
        self.parnames=None
        self.ncomps=None
        self.params=None
        self.errors=None
        self.rms=None
        self.residstd=None
        self.chisq=None
        self.dof=None
        self.redchisq=None
        self.AIC=None
        self.fitconverge=None

class saamodel(basemodel):
    """
    This houses the model information for spectral averaging areas. It uses
    the base model but includes some parameters that are unique to SAAs.

    Parameters
    ----------
    modeldict : dictionary
        This is a dictionary containing the model parameters that we want to
        add to the SAA. This is output from scousefitter.

    Attributes
    ----------
    SNR : number
        This is the signal-to-noise ratio set during the fitting process in
        scousefitter
    kernelsize : number
        This is the size of the kernel used for the derivative spectroscopy
        method
    manual : bool
        This indicates whether a manual fit was performed

    """
    def __init__(self, modeldict):
        super(basemodel, self).__init__()

        self.SNR=None
        self.kernelsize=None
        self.manual=None

        self.set_attributes(modeldict)

    def __repr__(self):
        """
        Return a nice printable format for the object.
        """
        return "< scousepy saamodel_solution; fittype={0}; ncomps={1} >".format(self.fittype, self.ncomps)

    def set_attributes(self, modeldict):
        """
        Sets the attributes of the SAA model
        """
        for parameter, value in modeldict.items():
            setattr(self, parameter, value)

class indivmodel(basemodel):
    """
    This houses the model information for individual spectra. It uses the base
    model and includes some parameters that are unique to individual spectra.

    Parameters
    ----------
    modeldict : dictionary
        This is a dictionary containing the model parameters that we want to
        add to the individual spectrum.

    """
    def __init__(self, modeldict):
        super(basemodel, self).__init__()

        self.method=None

        self.set_attributes(modeldict)

    def __repr__(self):
        """
        Return a nice printable format for the object.
        """
        return "< scousepy model_solution; fittype={0}; ncomps={1} >".format(self.fittype, self.ncomps)

    def set_attributes(self, modeldict):
        """
        Sets the attributes of the SAA model
        """
        for parameter, value in modeldict.items():
            setattr(self, parameter, value)
