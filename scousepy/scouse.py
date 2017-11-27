"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2017 Jonathan D. Henshaw
CONTACT: j.d.henshaw[AT]ljmu.ac.uk

"""

from __future__ import print_function

from astropy import units as u
from spectral_cube import SpectralCube
from astropy import wcs

from astropy import log
import numpy as np
import os
import sys
import warnings
import shutil
import time
import pyspeckit
warnings.simplefilter('ignore', wcs.FITSFixedWarning)

from .stage_1 import *
from .stage_2 import *
from .io import *
from .progressbar import AnimatedProgressBar

import matplotlib as mpl
import matplotlib.pyplot as plt

# add Python 2 xrange compatibility, to be removed
# later when we switch to numpy loops
if sys.version_info.major >= 3:
    range = range
else:
    range = xrange

try:
    input = raw_input
except NameError:
    pass

class scouse(object):

    def __init__(self):

        self.outputdirectory = None
        self.filename = None
        self.stagedirs = []
        self.cube = None
        self.rsaa = None
        self.ppv_vol = None
        self.rms_approx = None
        self.sigma_cut = None
        self.training_set = None
        self.sample_size = None
        self.saa_spectra = None
        self.coverage_coordinates = None
        self.completed_stages = []

    @staticmethod
    def stage_1(filename, datadirectory, ppv_vol, rsaa, rms_approx, sigma_cut, verbose = False, outputdir=None, write_moments=False, save_fig=True, training_set=False, samplesize=10):
        """
        Initial steps - here scousepy identifies the spatial area over which the
        fitting will be implemented.
        """

        self = scouse()
        self.filename = filename
        self.rsaa = rsaa
        self.ppv_vol = ppv_vol
        self.rms_approx = rms_approx
        self.sigma_cut = sigma_cut
        if training_set:
            self.training_set = True
            self.samplesize = samplesize
        else:
            self.training_set = False
            self.samplesize = 0

        starttime = time.time()
        # Generate file structure
        if outputdir==None:
            outputdir=datadirectory

        # directory structure
        fitsfile = os.path.join(datadirectory, self.filename+'.fits')
        self.outputdirectory = os.path.join(outputdir, filename)
        s1dir = os.path.join(outputdir, self.filename, 'stage_1')
        self.stagedirs.append(s1dir)

        # create the stage_1 directory
        mkdir_s1(self.outputdirectory, s1dir)

        if verbose:
            progress_bar = print_to_terminal(stage='s1', step='start')

        # Stop spectral cube from being noisy
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_log = log.level
            log.setLevel('ERROR')
            # Read in the datacube
            self.cube = SpectralCube.read(fitsfile).with_spectral_unit(u.km/u.s)
            # Generate moment maps
            momzero, momone, momtwo = get_moments(self, write_moments, s1dir, filename, verbose)
            # get the coverage / average the subcube spectra
            self.coverage_coordinates, self.saa_spectra = [], []
            for r in self.rsaa:
                cc, ss = define_coverage(self.cube, momzero.value, r, verbose)

                # for training_set implementation we want to randomly select
                # SAAs
                # TODO: current implementation can result in duplicates - fix
                if training_set:
                    cc, ss = get_random_saa(cc, ss, samplesize, r, verbose)
                else:
                    if verbose:
                        print('')
                _ss=np.array(ss)

                # Write a cube containing the average spectra. **DO NOT USE THIS
                # IN ANY WAY OTHER THAN TO EXTRACT THE DATA - POSITIONS WILL BE
                # ALL OFF DUE TO NYQUIST SAMPLING**
                write_averaged_spectra(self.cube.header, ss, r, s1dir)
                self.coverage_coordinates.append(cc)
                self.saa_spectra.append(ss)
            log.setLevel(old_log)

        if save_fig:
            # plot multiple coverage areas
            plot_rsaa(self.coverage_coordinates, momzero.value, self.rsaa, s1dir, filename)

        endtime = time.time()

        if verbose:
            progress_bar = print_to_terminal(stage='s1', step='end', length=np.size(momzero), var=cc, t1=starttime, t2=endtime)

        self.completed_stages.append('s1')
        return self

    def stage_2(self, model='gauss', verbose = False, training_set=False):
        """
        An interactive program designed to find best-fitting solutions to
        spatially averaged spectra taken from the SAAs.
        """

        s2dir = os.path.join(self.outputdirectory, 'stage_2')
        self.stagedirs.append(s2dir)
        # create the stage_2 directory
        mkdir_s2(self.outputdirectory, s2dir)

        if verbose:
            progress_bar = print_to_terminal(stage='s2', step='start')

        count=0
        # Generate the x_axis TODO: Add units
        x_axis = get_xaxis(self.cube.header)
        # Loop over potentially multiple rsaa values
        for i in range(len(self.rsaa)):
            # Generate an array containing spectra for only the current rsaa
            _saa_spectra = np.asarray(self.saa_spectra[int(i)])
            # Loop over the number of spectra
            for xind in range(np.shape(self.saa_spectra[int(i)])[2]):
                for yind in range(np.shape(self.saa_spectra[int(i)])[1]):
                    # Generate y axis TODO: Add units from cube header
                    y_axis = np.squeeze(_saa_spectra[:, yind, xind])
                    # TODO: pyspeckit needs a noise estimate - need a way of getting this for every spectrum ahead of the fitting process

                    # Some spectra are empty so skip these
                    if np.nansum(y_axis) != 0:
                        # Generate the spectrum for pyspeckit to fit
                        # Shhh
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            old_log = log.level
                            log.setLevel('ERROR')
                            # TODO: units! need to figure out what is going on w/units and if they can be specified easily from the header
                            spec = get_spec(self, x_axis, y_axis)
                            log.setLevel(old_log)

                        if training_set==True:
                            # TODO: If training_set = True - SAAs are randomly selected so you always want to perform interactive fitting
                            spec.plotter()
                            # TODO: Setting xmin and xmax doesn't seem to do anything?
                            # TODO: Can the interactive fitting be setup with user selected model as default?
                            # TODO: Can pyspeckit perform multi-component hfs fitting?!
                            spec.specfit(interactive=True, xmin=self.ppv_vol[0]*1000.0, xmax=self.ppv_vol[1]*1000.0, use_lmfit=True)
                            # TODO: Figure out the output of pyspeckit - Need to store this in a solution array
                            # TODO: Make the solution array as general as possible (dictionary perhaps?) so that the user can use different models
                            plt.show()
                        else:
                            # TODO: If training_set = False then the user has to fit all spectral averaging areas.
                            # TODO: Speed this up by using the previous best-fit as initial guess to current spectrum
                            # TODO: Can you give the user a choice in pyspeckit? i.e. show them the initial guess then let the user fit interactively if they aren't happy?
                            pass

                        count+=1
                        sys.exit()

        self.completed_stages.append('s2')
        return self


    def __repr__(self):
        """
        Return a nice printable format for the object.
        """

        return "<< scousepy object; stages_completed={} >>".format(self.completed_stages)
