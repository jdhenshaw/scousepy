# Licensed under an MIT open source license - see LICENSE
import numpy as np

class ScouseLearn(object):
    """
    Machine learning for decomposition
    """
    def __init__(self,
                 scouseobject=None):

        self.scouseobject=scouseobject
        self.xaxis=None
        self.spectra=None
        self.saa_dict=None
        self.unit=''
        self.xarrkwargs={}
        self.max_components=None
        self.trainingsetproperties=None
        self.n_examples=None
        self.random_seed=None

    def TrainingSetSAA(self, saa_dict, unit='',xarrkwargs={}):
        """
        Creates a training set from the saa spectra
        """

        self.saa_dict=saa_dict
        self.unit=unit
        self.xarrkwargs=xarrkwargs
        self.specx=self.scouseobject.xtrim
        self.spectra=get_spectraSAA(self)
        self.trainingsetproperties=get_properties(self)

    def make_training_set(self, max_components=1, trainingsetproperties={},
                          n_examples=200, random_seed=100):

        

def get_spectraSAA(self):
    """
    extracts a list of spectra from an SAA dictionary
    """
    return np.asarray([saa.spectrum[self.scouseobject.trimids]
                      for key, saa in self.saa_dict.items() if saa.to_be_fit])

def get_properties(self):
    """
    determines the properties of the data set. Specifically the distribution in
    noise, amplitude, velocity, and dispersion (determined using moments)

    """
    from scousepy.noisy import getnoise
    from scousepy.SpectralDecomposer import Decomposer

    rms=[getnoise(self.specx, spectrum).rms
         for i, spectrum in enumerate(self.spectra)]

    decomposers=[Decomposer(self.specx, spectrum, rms[i])
                 for i,spectrum in enumerate(self.spectra)]

    [Decomposer.create_a_spectrum(decomposer,unit=self.unit,xarrkwargs=self.xarrkwargs)
                                  for decomposer in decomposers]

    pskspectra=[decomposer.pskspectrum for decomposer in decomposers]

    moments=np.asarray([(pskspectrum.moments(unit=self.unit,vheight=False)[0:3])
                         for pskspectrum in pskspectra])

    stat_dict={'rms':rms,
               'amplitude':moments[:,0],
               'velocity': moments[:,1],
               'dispersion': moments[:,2]}

    trainingsetproperties={}
    for key in stat_dict.keys():
        stat=stat_dict[key]
        trainingsetproperties[key] = [np.min(stat), \
                                      np.median(stat),\
                                      np.max(stat)]

    return trainingsetproperties
