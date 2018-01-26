# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

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
import random
warnings.simplefilter('ignore', wcs.FITSFixedWarning)

from .stage_1 import *
from .stage_2 import *
from .stage_3 import *
from .io import *
from .progressbar import AnimatedProgressBar
from .saa_description import saa, add_ids, add_flat_ids
from .solution_description import fit

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
        self.saa_dict = None
        self.indiv_dict = None
        self.sample = None
        self.tolerances = None
        self.specres = None
        self.nrefine = None
        self.completed_stages = []

    @staticmethod
    def stage_1(filename, datadirectory, ppv_vol, rsaa, rms_approx, sigma_cut, \
                verbose = False, outputdir=None, write_moments=False, \
                save_fig=True, training_set=False, samplesize=10, \
                refine_grid=False, nrefine=3.0):
        """
        Initial steps - here scousepy identifies the spatial area over which the
        fitting will be implemented.
        """

        # TODO: Add optional refinement - base on peak vel versus mom1. refinement
        # SAA size where this value is high - statistical - increase refinement
        # based on deviation from mom1

        self = scouse()
        self.filename = filename
        self.rsaa = rsaa
        self.ppv_vol = ppv_vol
        self.rms_approx = rms_approx
        self.sigma_cut = sigma_cut
        self.nrefine = nrefine

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
            momzero, momone, momtwo, momnine = get_moments(self, write_moments, s1dir, filename, verbose)
            # get the coverage / average the subcube spectra
            self.saa_dict = {}

            # If the user has chosen to refine the grid
            if refine_grid:
                self.rsaa = get_rsaa(self)
                if verbose:
                    if np.size(self.rsaa) != self.nrefine:
                        raise ValueError('Rsaa < 1 pixel. Either increase Rsaa or decrease nrefine.')
                delta_v = calculate_delta_v(self, momone, momnine)
                # generate logarithmically spaced refinement steps
                step_values = generate_steps(self, delta_v)
                step_values.insert(0, 0.0)
            else:
                mom_zero = momzero.value

            for i, r in enumerate(self.rsaa, start=0):
                # Refine the mom zero grid if necessary
                self.saa_dict[i] = {}
                if refine_grid:
                    mom_zero = refine_momzero(self, momzero.value, delta_v, step_values[i], step_values[i+1])
                cc, ss, ids = define_coverage(self.cube, mom_zero, r, verbose)

                # Randomly select saas to be fit
                if training_set:
                    self.sample = get_random_saa(cc, samplesize, r, verbose=verbose)
                    totfit = len(self.sample)
                else:
                    self.sample = range(len(cc[:,0]))
                    totfit = len(cc[(np.isfinite(cc[:,0])),0])

                if verbose:
                    progress_bar = print_to_terminal(stage='s1', step='coverage', var=totfit)

                speccount=0
                for xind in range(np.shape(ss)[2]):
                    for yind in range(np.shape(ss)[1]):
                        sample = speccount in self.sample
                        SAA = saa(cc[speccount,:], ss[:, yind, xind],
                                     idx=speccount, sample = sample, \
                                     scouse=self)
                        self.saa_dict[i][speccount] = SAA
                        speccount+=1
                        indices = ids[SAA.index,(np.isfinite(ids[SAA.index,:,0])),:]
                        add_ids(SAA, indices)
                        add_flat_ids(SAA, scouse=self)

                if verbose:
                    print("")

            log.setLevel(old_log)

        if save_fig:
            # plot multiple coverage areas
            plot_rsaa(self.saa_dict, momzero.value, self.rsaa, s1dir, filename)

        endtime = time.time()

        if verbose:
            progress_bar = print_to_terminal(stage='s1', step='end', length=np.size(momzero), var=cc, t1=starttime, t2=endtime)

        self.completed_stages.append('s1')
        return self

    def stage_2(self, model='gauss', verbose = False, training_set=False,
                write_ascii=False):
        """
        An interactive program designed to find best-fitting solutions to
        spatially averaged spectra taken from the SAAs.
        """

        # TODO: Need to make this method more flexible - it would be good if the
        # user could fit the spectra in stages - minimise_tedium = True
        # TODO: Add an output option where the solutions are printed to file.
        # TODO: Allow for zero component fits

        s2dir = os.path.join(self.outputdirectory, 'stage_2')
        self.stagedirs.append(s2dir)
        # create the stage_2 directory
        mkdir_s2(self.outputdirectory, s2dir)

        if verbose:
            progress_bar = print_to_terminal(stage='s2', step='start')

        starttime = time.time()
        count=0

        # Cycle through potentially multiple Rsaa values
        for i in range(len(self.rsaa)):
            # Get the relavent SAA dictionary
            saa_dict = self.saa_dict[i]
            for j in range(len(saa_dict.keys())):
                # get the relavent SAA
                SAA = saa_dict[j]
                # If the SAA is to be fitted, pass it through the fitting
                # process
                if SAA.to_be_fit:
                    bf = fitting(self, SAA, training_set=training_set)
                    count+=1

            midtime=time.time()
            if verbose:
                progress_bar = print_to_terminal(stage='s2', step='mid', \
                                                 length=count, t1=starttime, \
                                                 t2=midtime)
        if write_ascii:
            output_ascii(self, s2dir)

        endtime = time.time()
        if verbose:
            progress_bar = print_to_terminal(stage='s2', step='end',
                                             t1=starttime, t2=endtime)

        self.completed_stages.append('s2')
        return self

    def stage_3(self, tol, \
                model='gaussian', verbose=False, training_set=False, \
                spatial=False, clear_cache = True):
        """
        This stage governs the automated fitting of the data
        """

        # TODO: Allow for zero component fits
        # TODO: Add spatial fitting methodolgy

        s3dir = os.path.join(self.outputdirectory, 'stage_3')
        self.stagedirs.append(s3dir)
        # create the stage_2 directory
        mkdir_s3(self.outputdirectory, s3dir)

        # initialise the dictionary containing all individual spectra
        self.indiv_dict = {}

        self.tolerances = np.array(tol)
        self.specres = self.cube.header['CDELT3']

        if verbose:
            progress_bar = print_to_terminal(stage='s3', step='start')

        # Begin by preparing the spectra and adding them to the relavent SAA
        initialise_indiv_spectra(self)
        # Fit the spectra
        fit_indiv_spectra(self, model=model, spatial=spatial)
        # Compile the spectra
        compile_spectra(self, spatial=spatial)
        # Clean things up a bit
        if clear_cache:
            clean_SAAs(self)

        self.completed_stages.append('s3')
        return self

    def __repr__(self):
        """
        Return a nice printable format for the object.
        """

        return "<< scousepy object; stages_completed={} >>".format(self.completed_stages)
