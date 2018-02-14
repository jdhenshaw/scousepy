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
from .stage_4 import *
from .stage_5 import interactive_plot
from .stage_6 import *
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
        self.saa_dict = None
        self.indiv_dict = None
        self.key_set = None
        self.sample = None
        self.tolerances = None
        self.specres = None
        self.nrefine = None
        self.fittype = None
        self.fitcount = 0.0
        self.blockcount = 0.0
        self.check_spec_indices = []
        self.completed_stages = []

    @staticmethod
    def stage_1(filename, datadirectory, ppv_vol, rsaa, rms_approx, sigma_cut, \
                verbose = False, outputdir=None, write_moments=False, \
                save_fig=True, training_set=False, samplesize=10, \
                refine_grid=False, nrefine=3.0, autosave=True, \
                fittype='gaussian'):
        """
        Initial steps - here scousepy identifies the spatial area over which the
        fitting will be implemented.
        """

        self = scouse()
        self.filename = filename
        self.datadirectory = datadirectory
        self.rsaa = rsaa
        self.ppv_vol = ppv_vol
        self.rms_approx = rms_approx
        self.sigma_cut = sigma_cut
        self.nrefine = nrefine
        self.fittype=fittype

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
            momzero, momone, momtwo, momnine = get_moments(self, write_moments,\
                                                           s1dir, filename,\
                                                           verbose)
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

            nref = self.nrefine
            for i, r in enumerate(self.rsaa, start=0):

                # Refine the mom zero grid if necessary
                self.saa_dict[i] = {}
                cc, ss, ids, frac = define_coverage(self.cube, momzero.value, \
                                                    momzero.value, r, 1.0, \
                                                    verbose)
                if refine_grid:
                    mom_zero = refine_momzero(self, momzero.value, delta_v, \
                                              step_values[i], step_values[i+1])
                    _cc, _ss, _ids, _frac = define_coverage(self.cube, \
                                                            momzero.value, \
                                                            mom_zero, r, nref, \
                                                            verbose, \
                                                            redefine=True)
                else:
                    _cc, _ss, _ids, _frape = cc, ss, ids, frac
                nref -= 1.0

                # Randomly select saas to be fit
                if self.training_set:
                    self.sample = get_random_saa(cc, samplesize, r, \
                                                 verbose=verbose)
                    totfit = len(self.sample)
                else:
                    if not refine_grid:
                        self.sample = range(len(cc[:,0]))
                        totfit = len(cc[(np.isfinite(cc[:,0])),0])
                    else:
                        self.sample = np.squeeze(np.where(np.isfinite(_cc[:,0])))
                        totfit = len(_cc[(np.isfinite(_cc[:,0])),0])

                if verbose:
                    progress_bar = print_to_terminal(stage='s1', \
                                                     step='coverage',var=totfit)

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
            log.setLevel(old_log)

        if save_fig:
            # plot multiple coverage areas
            plot_rsaa(self.saa_dict, momzero.value, self.rsaa, s1dir, filename)

        endtime = time.time()

        if verbose:
            progress_bar = print_to_terminal(stage='s1', step='end', \
                                             length=np.size(momzero), var=cc, \
                                             t1=starttime, t2=endtime)

        self.completed_stages.append('s1')

        # Save the scouse object automatically
        if autosave:
            self.save_to(self.datadirectory+self.filename+'/stage_1/s1.scousepy')

        return self

    def stage_2(self, verbose = False, write_ascii=False, autosave=True,
                fitrange=None):
        """
        An interactive program designed to find best-fitting solutions to
        spatially averaged spectra taken from the SAAs.
        """

        s2dir = os.path.join(self.outputdirectory, 'stage_2')
        self.stagedirs.append(s2dir)
        # create the stage_2 directory
        mkdir_s2(self.outputdirectory, s2dir)

        # generate a list of all SAA's (inc. all Rsaas)
        saa_list = generate_saa_list(self)
        saa_list = np.asarray(saa_list)

        if fitrange is not None:
            # Fail safe in case people try to re-run s1 midway through fitting
            # Without this - it would lose all previously fitted spectra.
            if np.min(fitrange) != 0.0:
                saa_dict=self.saa_dict[0]
                keys=list(saa_dict.keys())
                SAA=saa_dict[keys[0]]
                if SAA.model is None:
                    raise ValueError('DO NOT RE-RUN S1 - Load from autosaved S2 to avoid losing your work!')

        if verbose:
            progress_bar = print_to_terminal(stage='s2', step='start')

        starttime = time.time()

        # For staged fitting
        if fitrange==None:
            fitrange=np.arange(0,int(np.size(saa_list[:,0])))
        else:
            if np.max(fitrange)>np.size(saa_list):
                fitrange=np.arange(int(np.min(fitrange)),int(np.size(saa_list[:,0])))
            else:
                fitrange=np.arange(int(np.min(fitrange)),int(np.max(fitrange)))

        # Loop through the SAAs
        for i in fitrange:

            saa_dict = self.saa_dict[saa_list[i,1]]
            SAA = saa_dict[saa_list[i,0]]
            if SAA.index == 0.0:
                SAAid=0
                firstfit=True

            if SAA.to_be_fit:
                bf = fitting(self, SAA, saa_dict, SAAid, \
                             training_set=self.training_set, \
                             init_guess=firstfit)
                SAAid = SAA.index
                firstfit=False

            self.fitcount+=1

        if write_ascii and (self.fitcount == np.size(saa_list[:,0])):
            output_ascii_saa(self, s2dir)

        endtime = time.time()
        if verbose:
            progress_bar = print_to_terminal(stage='s2', step='end',
                                             t1=starttime, t2=endtime)

        self.completed_stages.append('s2')

        # Save the scouse object automatically
        if autosave:
            self.save_to(self.datadirectory+self.filename+'/stage_2/s2.scousepy')

        return self

    def stage_3(self, tol, njobs=1, verbose=False, \
                spatial=False, clear_cache=True, autosave=True):
        """
        This stage governs the automated fitting of the data
        """

        # TODO: Add spatial fitting methodolgy

        s3dir = os.path.join(self.outputdirectory, 'stage_3')
        self.stagedirs.append(s3dir)
        # create the stage_3 directory
        mkdir_s3(self.outputdirectory, s3dir)

        starttime = time.time()
        # initialise the dictionary containing all individual spectra
        indiv_dictionaries = {}

        self.tolerances = np.array(tol)
        self.specres = self.cube.header['CDELT3']

        if verbose:
            progress_bar = print_to_terminal(stage='s3', step='start')

        # Begin by preparing the spectra and adding them to the relavent SAA
        initialise_indiv_spectra(self)

        key_set = []
        # Cycle through potentially multiple Rsaa values
        for i in range(len(self.rsaa)):
            # Get the relavent SAA dictionary
            saa_dict = self.saa_dict[i]
            indiv_dictionaries[i] = {}
            # Fit the spectra
            fit_indiv_spectra(self, saa_dict, self.rsaa[i],\
                              njobs=njobs, spatial=spatial, verbose=verbose)
            # Compile the spectra
            indiv_dict = indiv_dictionaries[i]
            _key_set = compile_spectra(self, saa_dict, indiv_dict, self.rsaa[i], \
                                       spatial=spatial, verbose=verbose)
            # Clean things up a bit
            if clear_cache:
                clean_SAAs(self, saa_dict)
            key_set.append(_key_set)

        # At this stage there are multiple key sets - 1 for each rsaa value
        # compile into one.
        compile_key_sets(self, key_set)

        # merge multiple rsaa solutions into a single dictionary
        merge_dictionaries(self, indiv_dictionaries, \
                           spatial=spatial, verbose=verbose)
        # remove any duplicate entries
        remove_duplicates(self, verbose=verbose)

        endtime = time.time()
        if verbose:
            progress_bar = print_to_terminal(stage='s3', step='end', \
                                             t1=starttime, t2=endtime)

        self.completed_stages.append('s3')

        # Save the scouse object automatically
        if autosave:
            self.save_to(self.datadirectory+self.filename+'/stage_3/s3.scousepy')

        return self

    def stage_4(self, verbose=False, autosave=True):
        """
        In this stage we select the best fits out of those performed in stage 3.
        """

        s4dir = os.path.join(self.outputdirectory, 'stage_4')
        self.stagedirs.append(s4dir)
        # create the stage_4 directory
        mkdir_s4(self.outputdirectory, s4dir)

        starttime = time.time()

        if verbose:
            progress_bar = print_to_terminal(stage='s4', step='start')

        # select the best model out of those available - i.e. that with the
        # lowest aic value
        select_best_model(self)

        endtime = time.time()
        if verbose:
            progress_bar = print_to_terminal(stage='s4', step='end', \
                                             t1=starttime, t2=endtime)

        self.completed_stages.append('s4')

        # Save the scouse object automatically
        if autosave:
            self.save_to(self.datadirectory+self.filename+'/stage_4/s4.scousepy')

        return self

    def stage_5(self, blocksize = 6, figsize = None, plot_residuals=False, \
                verbose=False, autosave=True, blockrange=None, repeat=False,
                newfile=None):
        """
        In this stage the user is required to check the best-fitting solutions
        """

        s5dir = os.path.join(self.outputdirectory, 'stage_5')
        self.stagedirs.append(s5dir)
        # create the stage_5 directory
        mkdir_s5(self.outputdirectory, s5dir)

        if blockrange is not None:
            if repeat and (np.min(blockrange)==0.0):
                self.check_spec_indices=[]
            # Fail safe in case people try to re-run s1 midway through fitting
            # Without this - it would lose all previously fitted spectra.
            if np.min(blockrange) != 0.0:
                if np.size(self.check_spec_indices) == 0.0:
                    raise ValueError('Load from autosaved S5 to avoid losing your work!')

        starttime = time.time()

        if verbose:
            progress_bar = print_to_terminal(stage='s5', step='start')

        check_spec_indices = interactive_plot(self, blocksize, figsize,\
                                              plot_residuals=plot_residuals,\
                                              blockrange=blockrange)

        # For staged_checking - check and flatten
        self.check_spec_indices = check_and_flatten(self, check_spec_indices)

        endtime = time.time()
        if verbose:
            progress_bar = print_to_terminal(stage='s5', step='end', \
                                             t1=starttime, t2=endtime, \
                                             var=np.size(self.check_spec_indices))

        self.completed_stages.append('s5')

        # Save the scouse object automatically - create a backup if the user
        # wishes to iterate over s5 + s6
        if autosave:
            if repeat:
                if newfile is not None:
                    self.save_to(self.datadirectory+self.filename+newfile)
                else:
                    os.rename(self.datadirectory+self.filename+'/stage_5/s5.scousepy', \
                              self.datadirectory+self.filename+'/stage_5/s5.scousepy.bk')
                    self.save_to(self.datadirectory+self.filename+'/stage_5/s5.scousepy')
            else:
                self.save_to(self.datadirectory+self.filename+'/stage_5/s5.scousepy')

        return self

    def stage_6(self, plot_neighbours=False, radius_pix=1, figsize=[10,10], \
                plot_residuals=False, verbose=False, autosave=True, \
                write_ascii=False, specrange=None, repeat=None, newfile=None ):
        """
        In this stage the user takes a closer look at the spectra selected in s5
        """

        s6dir = os.path.join(self.outputdirectory, 'stage_6')
        self.stagedirs.append(s6dir)
        # create the stage_6 directory
        mkdir_s6(self.outputdirectory, s6dir)

        if specrange is not None:
            # Fail safe in case people try to re-run s1 midway through fitting
            # Without this - it would lose all previously fitted spectra.
            if np.min(specrange) != 0.0:
                if not 's6' in self.completed_stages:
                    raise ValueError('Load from autosaved S6 to avoid losing your work!')

        starttime = time.time()

        if verbose:
            progress_bar = print_to_terminal(stage='s6', step='start')

        # For staged refitting
        if specrange==None:
            specrange=np.arange(0,int(np.size(self.check_spec_indices)))
        else:
            if np.max(specrange)>int(np.size(self.check_spec_indices)):
                specrange=np.arange(int(np.min(specrange)),int(np.size(self.check_spec_indices)))
            else:
                specrange=np.arange(int(np.min(specrange)),int(np.max(specrange)))

        for i in specrange:
            key = self.check_spec_indices[i]
            if plot_neighbours:
                # Find the neighbours
                indices_adjacent = neighbours(np.shape(self.cube)[1:3], \
                                              int(key), radius_pix)
                # plot the neighbours
                plot_neighbour_pixels(self, indices_adjacent, figsize)

            models, selection = plot_alternatives(self, key, figsize, plot_residuals=plot_residuals)
            update_models(self, key, models, selection)

        if write_ascii:
            output_ascii_indiv(self, s6dir)

        endtime = time.time()
        if verbose:
            progress_bar = print_to_terminal(stage='s6', step='end', \
                                             t1=starttime, t2=endtime)
        # Save the scouse object automatically - create a backup if the user
        # wishes to iterate over s5 + s6
        if autosave:
            if repeat:
                if newfile is not None:
                    self.save_to(self.datadirectory+self.filename+newfile)
                else:
                    os.rename(self.datadirectory+self.filename+'/stage_6/s6.scousepy', \
                              self.datadirectory+self.filename+'/stage_6/s6.scousepy.bk')
                    self.save_to(self.datadirectory+self.filename+'/stage_6/s6.scousepy')
            else:
                self.save_to(self.datadirectory+self.filename+'/stage_6/s6.scousepy')
                
        self.completed_stages.append('s6')
        return self

    def __repr__(self):
        """
        Return a nice printable format for the object.
        """

        return "<< scousepy object; stages_completed={} >>".format(self.completed_stages)

#==============================================================================#
# io
#==============================================================================#

    def save_to(self, filename):
        """
        Saves an output file
        """
        from .io import save
        return save(self, filename)

    @staticmethod
    def load_from(filename):
        """
        Loads a previously computed scousepy file
        """
        from .io import load
        return load(filename)
