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
from .stage_5 import interactive_plot, DiagnosticImageFigure
from .stage_6 import *
from .io import *
from .progressbar import AnimatedProgressBar
from .saa_description import saa, add_ids
from .solution_description import fit

import matplotlib as mpl
import matplotlib.pyplot as plt

Fitter = Stage2Fitter()
fitting = Fitter.fitting

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

    def __init__(self, filename=None, outputdir=None, fittype=None,
                 datadirectory=None):

        self.filename = filename
        self.datadirectory = datadirectory
        if outputdir is not None:
            self.outputdirectory = os.path.join(outputdir, filename)
        self.stagedirs = []
        self.cube = None
        self.wsaa = None
        self.ppv_vol = None
        self.rms_approx = None
        self.mask_below = 0.0
        self.training_set = None
        self.sample_size = None
        self.tolerances = None
        self.specres = None
        self.nrefine = None
        self.fittype = fittype
        self.sample = None
        self.x = None
        self.xtrim = None
        self.trimids=None
        self.saa_dict = None
        self.indiv_dict = None
        self.key_set = None
        self.fitcount = 0
        self.blockcount = 0.0
        self.check_spec_indices = []
        self.check_block_indices = []
        self.blocksize = None
        self.completed_stages = []

    def load_cube(self, fitsfile=None, cube=None):
        """
        Load in a cube

        Parameters
        ----------
        fitsfile : fits
            File in fits format to be read in
        cube : spectral cube
            If fits file is not supplied - provide a spectral cube object
            instead

        """

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_log = log.level
            log.setLevel('ERROR')

            # Read in the datacube
            if cube is None:
                self.cube = SpectralCube.read(fitsfile).with_spectral_unit(u.km/u.s,
                                                                           velocity_convention='radio')
            else:
                self.cube = cube

            if self.cube.spectral_axis.diff()[0] < 0:
                if np.abs(self.cube.spectral_axis[0].value - self.cube[::-1].spectral_axis[-1].value) > 1e-5:
                    raise ImportError("Update to a more recent version of spectral-cube "
                                      " or reverse the axes manually.")
                self.cube = self.cube[::-1]

            # Generate the x axis common to the fitting process
            self.x, self.xtrim, self.trimids = get_x_axis(self)
            # Compute typical noise within the spectra
            self.rms_approx = compute_noise(self)

    @staticmethod
    def stage_1(filename, datadirectory, ppv_vol, wsaa, mask_below=0.0,
                cube=None, verbose = False, outputdir=None,
                write_moments=False, save_fig=True, training_set=False,
                samplesize=10, refine_grid=False, nrefine=3.0, autosave=True,
                fittype='gaussian'):
        """
        Initial steps - here scousepy identifies the spatial area over which the
        fitting will be implemented.

        Parameters
        ----------
        filename : string
            Name of the file to be loaded
        datadirectory : string
            Directory containing the datacube
        ppv_vol : list
            A list containing boundaries for fitting. You can use this to
            selectively fit part of a datacube. Should be in the format
            ppv_vol = [vmin, vmax, ymin, ymax, xmin, xmax] with the velocities
            in absolute units and the x, y values in pixels. If all are set to
            zero scouse will ignore this and just fit the whole cube.
        wsaa : list
            The width of a spectral averaging area in pixels. Note this has
            been updated from the IDL implementation where it previously used a
            half-width (denoted rsaa). Can provide multiple values in a list
            as an alternative to the refine_grid option (see below).
        mask_below : float, optional
            Used for moment computation - mask all data below this absolute
            value.
        cube : spectral cube object, optional
            Load in a spectral cube rather than a fits file.
        verbose : bool, optional
            Verbose output to terminal
        outputdir : string, optional
            Alternate output directory. Deflault is datadirectory
        write_moments : bool, optional
            If true, scouse will write fits files of the moment 0, 1, and 2 as
            well as the moment 9 (casa notation - velocity channel of peak
            emission).
        save_fig : bool, optional
            If true, scouse will output a figure of the coverage
        training_set : bool, optional
            Can be used in combination with samplesize (see below). If true,
            scouse will select SAAs at random for use as a training set. These
            can be fit as normal and the solutions supplied to machine learning
            algorithms for the fitting of very large data cubes.
        sample_size : float, optional
            The number of SAAs that will make up your training set.
        refine_grid : bool, optional
            If true, scouse will refine the SAA size.
        nrefine : float, optional
            The number of refinements of the SAA size.
        autosave : bool, optional
            Save the output at each stage of the process.
        fittype : string
            Compatible with pyspeckit's models for fitting different types of
            models. Defualt is Gaussian fitting.
        """

        if outputdir is None:
            outputdir=datadirectory
        self = scouse(fittype=fittype, filename=filename, outputdir=outputdir, datadirectory=datadirectory)
        self.wsaa = wsaa
        self.ppv_vol = ppv_vol
        self.nrefine = nrefine
        self.mask_below=mask_below

        if training_set:
            self.training_set = True
            self.samplesize = samplesize
        else:
            self.training_set = False
            self.samplesize = 0

        # Main routine
        starttime = time.time()

        # directory structure
        fitsfile = os.path.join(datadirectory, self.filename+'.fits')
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

            self.load_cube(fitsfile=fitsfile)

            # Generate moment maps
            momzero, momone, momtwo, momnine = get_moments(self, write_moments,
                                                           s1dir, filename,
                                                           verbose)

            # get the coverage / average the subcube spectra
            self.saa_dict = {}

            # If the user has chosen to refine the grid
            if refine_grid:
                self.wsaa = get_wsaa(self)
                if verbose:
                    if np.size(self.wsaa) != self.nrefine:
                        raise ValueError('wsaa < 1 pixel. Either increase wsaa or decrease nrefine.')

                delta_v = calculate_delta_v(self, momone, momnine)
                # generate logarithmically spaced refinement steps
                step_values = generate_steps(self, delta_v)
                step_values.insert(0, 0.0)
            else:
                mom_zero = momzero.value

            nref = self.nrefine
            for i, w in enumerate(self.wsaa, start=0):
                # Create a dictionary to house the SAAs
                self.saa_dict[i] = {}

                # Make a first pass at defining the coverage.
                cc, ss, ids, frac = define_coverage(self.cube, momzero.value,
                                                    momzero.value, w, 1.0,
                                                    verbose)

                if refine_grid:
                    # When refining the coverage - we have to recompute the
                    # momzero map according to which regions have more complex
                    # line profiles. As such we need to recompute _cc, _ids, and
                    # _frac. _ss will be the same (the spectra don't change)
                    # and so these are not recomputed (see line 264 in stage_1).
                    # However, we do want to know which coverage boxes to retain

                    mom_zero = refine_momzero(self, momzero.value, delta_v,
                                              step_values[i], step_values[i+1])
                    _cc, _ss, _ids, _frac = define_coverage(self.cube,
                                                            momzero.value,
                                                            mom_zero, w, nref,
                                                            verbose,
                                                            redefine=True)
                else:
                    _cc, _ss, _ids, _frac = cc, ss, ids, frac
                nref -= 1.0

                if self.training_set:
                    # Randomly select saas to be fit
                    self.sample = get_random_saa(cc, samplesize, r,
                                                 verbose=verbose)
                    totfit = len(self.sample)
                else:
                    if not refine_grid:
                        # Define the sample of spectra to fit - i.e. where cc
                        # is finite
                        self.sample = np.squeeze(np.where(np.isfinite(cc[:,0])))
                        totfit = len(cc[(np.isfinite(cc[:,0])),0])
                    else:
                        # If refining the grid use _cc as well - i.e. the
                        # recomputed positions based on the refined momzero map
                        self.sample = np.squeeze(np.where(np.isfinite(_cc[:,0])))
                        totfit = len(_cc[(np.isfinite(_cc[:,0])),0])

                if verbose:
                    progress_bar = print_to_terminal(stage='s1',
                                                     step='coverage',
                                                     var=totfit)

                speccount=0
                # Now cycle through the spatially-averaged spectra
                for xind in range(np.shape(ss)[2]):
                    for yind in range(np.shape(ss)[1]):
                        # Every SAA gets a spectrum even if it is not to be
                        # fitted - this is probably a bit wasteful. If the
                        # spectrum is contained within the sample (see above)
                        # it will be fitted during stage 2.
                        sample = speccount in self.sample
                        # generate the SAA
                        SAA = saa(cc[speccount,:], ss[:, yind, xind],
                                  idx=speccount, sample=sample, scouse=self)
                        # Add the SAA to the dictionary
                        self.saa_dict[i][speccount] = SAA
                        # Add the indices of the individual spectra contained
                        # within the SAA box to the SAA.
                        indices = ids[SAA.index,(np.isfinite(ids[SAA.index,:,0])),:]
                        add_ids(SAA, indices)
                        speccount+=1
            log.setLevel(old_log)

        if save_fig:
            # plot multiple coverage areas
            plot_wsaa(self.saa_dict, momzero.value, self.wsaa, s1dir, filename)

        endtime = time.time()

        if verbose:
            progress_bar = print_to_terminal(stage='s1', step='end',
                                             length=np.size(momzero), var=cc,
                                             t1=starttime, t2=endtime)

        self.completed_stages.append('s1')

        # Save the scouse object automatically
        if autosave:
            with open(self.datadirectory+self.filename+'/stage_1/s1.scousepy', 'wb') as fh:
                pickle.dump((self.saa_dict, self.wsaa, self.ppv_vol), fh)

        input("Press enter to continue...")
        # close all figures before moving on
        # (only needed for plt.ion() case)
        plt.close('all')

        return self

    def load_stage_1(self, fn):
        with open(fn, 'rb') as fh:
            self.saa_dict,self.wsaa, self.ppv_vol = pickle.load(fh)
        self.completed_stages.append('s1')

    def stage_2(self, verbose = False, write_ascii=False, autosave=True,
                staged=False, nspec=None):
        """
        An interactive program designed to find best-fitting solutions to
        spatially averaged spectra taken from the SAAs.

        Parameters
        ----------
        verbose : bool, optional
            Verbose output of fitting process.
        write_ascii : bool, optional
            Outputs an ascii table containing the best fitting solutions to the
            spectral averaging areas.
        autosave : bool, optional
            Autosaves the scouse file.
        staged : bool, optional
            Staged fitting. Allows a user to break the fitting process down into
            multiple stages. Combined with nspec a user can fit 'nspec' spectra
            at a time.

        """

        s2dir = os.path.join(self.outputdirectory, 'stage_2')
        self.stagedirs.append(s2dir)
        # create the stage_2 directory
        mkdir_s2(self.outputdirectory, s2dir)

        # generate a list of all SAA's (inc. all wsaas)
        saa_list = generate_saa_list(self)
        saa_list = np.asarray(saa_list)

        # Staged fitting preparation
        if staged:
            if self.fitcount != 0.0:
                lower = int(self.fitcount)
                upper = int(lower+nspec)
            else:
                lower = 0
                upper = int(lower+nspec)

            # Fail safe in case people try to re-run s1 midway through fitting
            # Without this - it would lose all previously fitted spectra.
            if lower != 0:
                saa_dict=self.saa_dict[0]
                keys=list(saa_dict.keys())
                cont=True
                counter=0
                while cont:
                    key = keys[counter]
                    SAA=saa_dict[key]
                    if SAA.to_be_fit:
                        if SAA.model is None:
                            raise ValueError('DO NOT RE-RUN S1 - Load from autosaved S2 to avoid losing your work!')
                        else:
                            cont=False
                    else:
                        counter+=1

        if verbose:
            progress_bar = print_to_terminal(stage='s2', step='start')

        starttime = time.time()

        # Set ranges for staged fitting
        if not staged:
            fitrange=np.arange(0,int(np.size(saa_list[:,0])))
        else:
            if upper>=np.size(saa_list[:,0]):
                if lower >= np.size(saa_list[:,0]):
                    fitrange=[]
                else:
                    fitrange=np.arange(int(lower),int(np.size(saa_list[:,0])))
            else:
                fitrange=np.arange(int(lower),int(upper))

        # determine how many fits we will actually be performing
        n_to_fit = sum([self.saa_dict[saa_list[ii,1]][saa_list[ii,0]].to_be_fit
                        for ii in fitrange])

        if n_to_fit <= 0:
            raise ValueError("No spectra are selected to be fit.")

        # Loop through the SAAs
        for i_,i in enumerate(fitrange):
            print("Fitting {0} out of {1}".format(i_+1, n_to_fit))

            # Get the relevant SAA dictionary (if multiple wsaa values are
            # supplied)
            saa_dict = self.saa_dict[saa_list[i,1]]
            # Get the first SAA to fit
            SAA = saa_dict[saa_list[i,0]]

            # Fitting process is different for the first SAA in a wsaa loop.
            # For all subsequent SAAs scouse will try and apply the previous
            # solution to the spectrum in an attempt to speed things up and
            # reduce the interactivity
            if SAA.index == 0.0:
                SAAid=0
                firstfit=True
            elif i == np.min(fitrange):
                SAAid=SAA.index
                firstfit=True

            if SAA.to_be_fit:
                with warnings.catch_warnings():
                    # This is to catch an annoying matplotlib deprecation warning:
                    # "Using default event loop until function specific to this GUI is implemented"
                    warnings.simplefilter('ignore', category=DeprecationWarning)

                    # enter the fitting process
                    bf = fitting(self, SAA, saa_dict, SAAid,
                                 training_set=self.training_set,
                                 init_guess=firstfit)
                SAAid = SAA.index
                firstfit=False

            self.fitcount+=1

        # Output at the end of SAA fitting
        if write_ascii and (self.fitcount == np.size(saa_list[:,0])):
            output_ascii_saa(self, s2dir)
            self.completed_stages.append('s2')

        endtime = time.time()
        if verbose:
            progress_bar = print_to_terminal(stage='s2', step='end',
                                             t1=starttime, t2=endtime)

        # Save the scouse object automatically
        if autosave:
            with open(self.datadirectory+self.filename+'/stage_2/s2.scousepy', 'wb') as fh:
                pickle.dump(self.saa_dict, fh)

        # close all figures before moving on
        # (only needed for plt.ion() case)
        plt.close('all')

        return self

    def load_stage_2(self, fn):
        with open(fn, 'rb') as fh:
            self.saa_dict = pickle.load(fh)
        self.completed_stages.append('s2')

    def stage_3(self, tol, njobs=1, verbose=False, spatial=False,
                clear_cache=True, autosave=True):
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

        # Begin by preparing the spectra and adding them to the relevant SAA
        initialise_indiv_spectra(self, verbose=verbose, njobs=njobs)

        key_set = []
        # Cycle through potentially multiple wsaa values
        for i in range(len(self.wsaa)):
            # Get the relavent SAA dictionary
            saa_dict = self.saa_dict[i]
            indiv_dictionaries[i] = {}
            # Fit the spectra
            fit_indiv_spectra(self, saa_dict, self.wsaa[i], njobs=njobs,
                              spatial=spatial, verbose=verbose)


            # Compile the spectra
            indiv_dict = indiv_dictionaries[i]
            _key_set = compile_spectra(self, saa_dict, indiv_dict,
                                       self.wsaa[i], spatial=spatial,
                                       verbose=verbose)
            # Clean things up a bit
            if clear_cache:
                clean_SAAs(self, saa_dict)
            key_set.append(_key_set)


        # At this stage there are multiple key sets: 1 for each wsaa value
        # compile into one.
        compile_key_sets(self, key_set)
        # merge multiple wsaa solutions into a single dictionary
        merge_dictionaries(self, indiv_dictionaries,
                           spatial=spatial, verbose=verbose)
        # remove any duplicate entries
        remove_duplicates(self, verbose=verbose)

        endtime = time.time()
        if verbose:
            progress_bar = print_to_terminal(stage='s3', step='end',
                                             t1=starttime, t2=endtime)

        self.completed_stages.append('s3')

        # Save the scouse object automatically
        if autosave:
            with open(self.datadirectory+self.filename+'/stage_3/s3.scousepy', 'wb') as fh:
                pickle.dump((self.indiv_dict, self.tolerances), fh)

        return self

    def load_indiv_dicts(self, fn, stage):
        with open(fn, 'rb') as fh:
            self.indiv_dict = pickle.load(fh)
        self.completed_stages.append(stage)

    def load_stage_3(self, fn):
        with open(fn, 'rb') as fh:
            self.indiv_dict, self.tolerances = pickle.load(fh)
        self.completed_stages.append('s3')

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
            progress_bar = print_to_terminal(stage='s4', step='end',
                                             t1=starttime, t2=endtime)

        self.completed_stages.append('s4')

        # Save the scouse object automatically
        if autosave:
            with open(self.datadirectory+self.filename+'/stage_4/s4.scousepy', 'wb') as fh:
                pickle.dump(self.indiv_dict, fh)

        return self

    def load_stage_4(self, fn):
        return self.load_indiv_dicts(fn, stage='s4')

    def stage_5(self, blocksize = 6, figsize = None, plot_residuals=False,
                verbose=False, autosave=True, blockrange=None, repeat=False,
                newfile=None):
        """
        In this stage the user is required to check the best-fitting solutions
        """
        self.blocksize = blocksize

        s5dir = os.path.join(self.outputdirectory, 'stage_5')
        self.stagedirs.append(s5dir)
        # create the stage_5 directory
        mkdir_s5(self.outputdirectory, s5dir)

        interactive_state = plt.matplotlib.rcParams['interactive']
        plt.ion()

        dd = DiagnosticImageFigure(self, blocksize=blocksize, savedir=s5dir)

        dd.show_first()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            while not dd.done:
                try:
                    # using just a few little bits of plt.pause below
                    dd.fig.canvas.draw()
                    dd.fig.canvas.start_event_loop(0.1)
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    break

        plt.matplotlib.rcParams['interactive'] = interactive_state

        check_spec_indices = dd.check_spec_indices
        check_block_indices = dd.check_block_indices

        #if blockrange is not None:
        #    if repeat and (np.min(blockrange)==0.0):
        #        self.check_spec_indices=[]
        #    # Fail safe in case people try to re-run s1 midway through fitting
        #    # Without this - it would lose all previously fitted spectra.
        #    if np.min(blockrange) != 0.0:
        #        if np.size(self.check_spec_indices) == 0.0:
        #            raise ValueError('Load from autosaved S5 to avoid losing your work!')

        #starttime = time.time()

        #if verbose:
        #    progress_bar = print_to_terminal(stage='s5', step='start')


        ## interactive must be forced to 'false' for this section to work
        #interactive_state = plt.matplotlib.rcParams['interactive']
        ##plt.ioff()
        #check_spec_indices = interactive_plot(self, blocksize, figsize,\
        #                                      plot_residuals=plot_residuals,\
        #                                      blockrange=blockrange)
        #plt.matplotlib.rcParams['interactive'] = interactive_state

        # For staged_checking - check and flatten
        self.check_spec_indices, self.check_block_indices = check_and_flatten(self, check_spec_indices, check_block_indices)
        self.check_spec_indices = np.asarray(self.check_spec_indices)
        self.check_block_indices = np.asarray(self.check_block_indices)

        #endtime = time.time()
        #if verbose:
        #    progress_bar = print_to_terminal(stage='s5', step='end', \
        #                                     t1=starttime, t2=endtime, \
        #                                     var=np.size(self.check_spec_indices))

        self.completed_stages.append('s5')

        # Save the scouse object automatically - create a backup if the user
        # wishes to iterate over s5 + s6
        if autosave:
            with open(self.datadirectory+self.filename+'/stage_5/s5.scousepy', 'wb') as fh:
                pickle.dump((self.check_spec_indices, self.check_block_indices, self.blocksize), fh)

        return self

    def load_stage_5(self, fn):
        with open(self.datadirectory+self.filename+'/stage_3/s3.scousepy', 'rb') as fh:
            val, self.tolerances = pickle.load(fh)
        with open(fn, 'rb') as fh:
            self.check_spec_indices, self.check_block_indices, self.blocksize = pickle.load(fh)
        self.completed_stages.append('s5')

    def stage_6(self, plot_neighbours=False, radius_pix=1, figsize=[10,10],
                plot_residuals=False, verbose=False, autosave=True,
                write_ascii=False, specrange=None, repeat=None, newfile=None,
                njobs=1 ):
        """
        In this stage the user takes a closer look at the spectra selected in s5
        """

        # temporary fix: eventually, this should look like stage 2, with
        # interactive figures
        interactive_state = plt.matplotlib.rcParams['interactive']
        plt.ion()

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

        # Firstly check the check_spec_indices against the blocks and remove any
        # duplicates
        self.check_spec_indices = check_blocks(self)

        # For staged refitting
        if specrange is None:
            specrange=np.arange(0,int(np.size(self.check_spec_indices)))
        else:
            if np.max(specrange)>int(np.size(self.check_spec_indices)):
                specrange=np.arange(int(np.min(specrange)),int(np.size(self.check_spec_indices)))
            else:
                specrange=np.arange(int(np.min(specrange)),int(np.max(specrange)))

        # Here we will fit the individual spectra that have been selected for
        # refitting
        for i in specrange:
            key = self.check_spec_indices[i]

            # This first of all plots the neighbouring pixels - this can be
            # useful if you forget why you selected that spectrum in the first
            # place - it helps to provide a bit of context
            if plot_neighbours:
                # Find the neighbours
                indices_adjacent = neighbours(np.shape(self.cube)[1:3],
                                              int(key), radius_pix)
                # plot the neighbours
                plot_neighbour_pixels(self, indices_adjacent, figsize)

            # This will plot the current model solution as well as all possible
            # alternatives. The user should either select one of these or
            # press enter to enter the manual fitting mode
            models, selection = plot_alternatives(self, key, figsize, plot_residuals=plot_residuals)
            update_models(self, key, models, selection)

        # Stage 5 gives the user the option to select all pixels within a block
        # for refitting - this first of all generates a pseudo-SAA from the
        # block, we then manually fit. This solution is then applied to all
        # spectra contained within the block.
        block_dict={}
        # cycle through all the blocks
        for blocknum in self.check_block_indices:
            # create an empty spectrum
            spec = np.zeros(self.cube.shape[0])
            # get all of the individual pixel indices contained within that
            # block
            block_indices = get_block_indices(self, blocknum)
            # turn the flattened indices into 2D indices such that we can find
            # the spectra in the cube
            coords = gen_2d_coords(self,block_indices)
            # create an SAA
            SAA = gen_pseudo_SAA(self, coords, block_dict, blocknum, spec)
            # prepare the spectra for fitting
            initialise_indiv_spectra_s6(self, SAA, njobs)
            # Manual fitting of the blocks
            manually_fit_blocks(self, block_dict, blocknum)
        # automated fitting of block spectra
        auto_fit_blocks(self, block_dict, njobs, self.blocksize, verbose=verbose)

        if write_ascii:
            output_ascii_indiv(self, s6dir)

        endtime = time.time()
        if verbose:
            progress_bar = print_to_terminal(stage='s6', step='end',
                                             t1=starttime, t2=endtime)

        if autosave:
            with open(self.datadirectory+self.filename+'/stage_6/s6.scousepy', 'wb') as fh:
                pickle.dump(self.indiv_dict, fh)

        self.completed_stages.append('s6')

        # reset the interactive state to whatever it was before
        plt.matplotlib.rcParams['interactive'] = interactive_state

        return self

    def load_stage_6(self, fn):
        return self.load_indiv_dicts(fn, stage='s6')

    def __repr__(self):
        """
        Return a nice printable format for the object.
        """

        return "< scousepy object; stages_completed={} >".format(self.completed_stages)

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

#==============================================================================#
# Analysis
#==============================================================================#
    @staticmethod
    def compute_stats(self):
        """
        Computes some statistics for the fitting process
        """
        from .statistics import stats
        return stats(scouse=self)
