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
from .best_fitting_solution import fit

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

        starttime = time.time()
        # Create an empty dictionary to hold the best-fitting solutions
        self.saa_fits = {}

        count=0
        # Generate x axis - trim according to user inputs
        x_axis_notrim = get_xaxis(self)
        keep = ((x_axis_notrim>self.ppv_vol[0])&(x_axis_notrim<self.ppv_vol[1]))
        #print(keep)
        x_axis = x_axis_notrim[keep]
        # Loop over potentially multiple rsaa values
        for i in range(len(self.rsaa)):
            self.saa_fits[i] = {}
            # Generate an array containing spectra for only the current rsaa
            _saa_spectra = np.asarray(self.saa_spectra[int(i)])
            # Loop over the number of spectra
            for xind in range(np.shape(self.saa_spectra[int(i)])[2]):
                for yind in range(np.shape(self.saa_spectra[int(i)])[1]):
                    # Trim flux data
                    y_axis_notrim = np.squeeze(_saa_spectra[:, yind, xind])
                    y_axis = y_axis_notrim[keep]
                    # Some spectra are empty so skip these
                    if np.nansum(y_axis) != 0:
                        # Establish noise
                        rmsnoise = get_noise(self, x_axis_notrim, y_axis_notrim)

                        if training_set==True:
                            # If training_set=True, need to cycle through all of
                            # the sample spectra and fit manually.
                            bf = fitting(self, i, x_axis, y_axis, rmsnoise, count, training_set=True)
                        else:
                            bf = fitting(self, i, x_axis, y_axis, rmsnoise, count, training_set=False)
                        count+=1

            midtime=time.time()
            if verbose:
                progress_bar = print_to_terminal(stage='s2', step='mid', length=count, t1=starttime, t2=midtime)

        endtime = time.time()
        if verbose:
            progress_bar = print_to_terminal(stage='s2', step='end', t1=starttime, t2=endtime)

        self.completed_stages.append('s2')
        return self

    def stage_3(self, verbose=False, training_set=False):

        self.completed_stages.append('s3')
        return self
    def __repr__(self):
        """
        Return a nice printable format for the object.
        """

        return "<< scousepy object; stages_completed={} >>".format(self.completed_stages)
