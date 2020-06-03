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
from .saa_description import saa, add_ids
from .solution_description import fit
from .colors import *

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

Fitter = Stage2Fitter()
fitting = Fitter.preparefit

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
        self.fitcounts6 = 0
        self.blockcount = 0
        self.blocksize = None
        self.check_spec_indices = []
        self.check_block_indices = []
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
                _cube = SpectralCube.read(fitsfile).with_spectral_unit(u.km/u.s,
                                                    velocity_convention='radio')
            else:
                _cube = cube

            if _cube.spectral_axis.diff()[0] < 0:
                if np.abs(_cube.spectral_axis[0].value -
                                    _cube[::-1].spectral_axis[-1].value) > 1e-5:
                    raise ImportError("Update to a more recent version of "
                                      "spectral-cube or reverse the axes "
                                      "manually.")
                _cube = _cube[::-1]

            # Trim cube if necessary
            if (self.ppv_vol[2] is not None) & (self.ppv_vol[3] is not None):
                _cube = _cube[:, int(self.ppv_vol[2]):int(self.ppv_vol[3]), :]
            if (self.ppv_vol[4] is not None) & (self.ppv_vol[5] is not None):
                _cube = _cube[:, :, int(self.ppv_vol[4]):int(self.ppv_vol[5])]

            self.cube = _cube
            # Generate the x axis common to the fitting process
            self.x, self.xtrim, self.trimids = get_x_axis(self)
            # Compute typical noise within the spectra
            self.rms_approx = compute_noise(self)

    @staticmethod
    def stage_1(filename, datadirectory,
                wsaa, ppv_vol=[None, None, None, None, None, None],
                mask_below=0.0, cube=None, verbose = False, outputdir=None,
                write_moments=False, save_fig=True, training_set=False,
                samplesize=10, refine_grid=False, nrefine=3.0, autosave=True,
                fittype='gaussian'):
        """
        Stage 1

        Identify the spatial area over which the fitting will be implemented.

        Parameters
        ----------
        filename : string
            Name of the file to be loaded
        datadirectory : string
            Directory containing the datacube
        wsaa : number
            The width of a spectral averaging area in pixels. Note this has
            been updated from the IDL implementation where it previously used a
            half-width (denoted rsaa). Can provide multiple values in a list
            as an alternative to the refine_grid option (see below).
        ppv_vol : array like, optional
            A list containing boundaries for fitting. You can use this to
            selectively fit part of a datacube. Should be in the format
            ppv_vol = [vmin, vmax, ymin, ymax, xmin, xmax] with the velocities
            in absolute units and the x, y values in pixels. If all are set to
            None scouse will ignore this and just fit the whole cube. Default
            is ppv_vol = [None, None, None, None, None, None]; whole cube is
            fitted.
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

        self = scouse(fittype=fittype, filename=filename, outputdir=outputdir,
                      datadirectory=datadirectory)
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
                        raise ValueError(colors.fg._red_+"wsaa < 1 pixel. "+
                                         "Either increase wsaa or decrease"+
                                         " nrefine."+colors.fg._endc_)

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
                                                    momzero.value, w, nrefine,
                                                    verbose,
                                                    refine_grid=refine_grid,
                                                    redefine=False)
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
                                                            refine_grid=refine_grid,
                                                            redefine=True)
                else:
                    _cc, _ss, _ids, _frac = cc, ss, ids, frac
                nref -= 1.0

                if self.training_set:
                    # Randomly select saas to be fit
                    self.sample = get_random_saa(cc, samplesize, w,
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
                        indices = ids[SAA.index,
                                            (np.isfinite(ids[SAA.index,:,0])),:]
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
            with open(self.outputdirectory+'/stage_1/s1.scousepy', 'wb') as fh:
                pickle.dump((self.saa_dict, self.wsaa, self.ppv_vol,
                                                      self.outputdirectory), fh)

        input("Press enter to continue...")
        # close all figures before moving on
        # (only needed for plt.ion() case)
        plt.close('all')

        return self

    def load_stage_1(self, fn):
        with open(fn, 'rb') as fh:
            self.saa_dict,self.wsaa,self.ppv_vol,self.outputdirectory = \
                                                                 pickle.load(fh)
        self.completed_stages.append('s1')

    def stage_2(self, verbose = False, write_ascii=False, autosave=True,
                bitesize=False, nspec=None, training_set=False):
        """
        Stage 2

        Manual fitting of the SAAs

        Parameters
        ----------
        verbose : bool, optional
            Verbose output of fitting process.
        write_ascii : bool, optional
            Outputs an ascii table containing the best fitting solutions to the
            spectral averaging areas.
        autosave : bool, optional
            Autosaves the scouse file.
        bitesize : bool, optional
            Bitesized fitting. Allows a user to break the fitting process down
            into multiple stages. Combined with nspec a user can fit 'nspec'
            spectra at a time. For large data cubes fitting everything in one go
            can be a bit much...
        nspec : int, optional
            Fit this many spectra at a time.

        """

        s2dir = os.path.join(self.outputdirectory, 'stage_2')
        self.stagedirs.append(s2dir)
        # create the stage_2 directory
        mkdir_s2(self.outputdirectory, s2dir)

        # generate a list of all SAA's (inc. all wsaas)
        saa_list = generate_saa_list(self)
        saa_list = np.asarray(saa_list)

        # bitesize fitting preparation
        if bitesize:
            if self.fitcount != 0.0:
                lower = int(self.fitcount)
                upper = int(lower+nspec)
            else:
                lower = 0
                upper = int(lower+nspec)

        if verbose:
            progress_bar = print_to_terminal(stage='s2', step='start')

        starttime = time.time()

        # Set ranges for bitesize fitting
        if not bitesize:
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
            raise ValueError(colors.fg._red_+"No spectra are selected to be fit."+
                             "Fitting has completed."+colors._endc_)

        # Loop through the SAAs
        for i_,i in enumerate(fitrange):
            print(colors.fg._lightgrey_+"====================================================="+colors._endc_)
            print(colors.fg._lightgrey_+"Fitting {0} out of {1}".format(i_+1, n_to_fit)+colors._endc_)
            print(colors.fg._lightgrey_+"====================================================="+colors._endc_)
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
            elif training_set:
                SAAid=SAA.index
                firstfit=True

            if SAA.to_be_fit:
                with warnings.catch_warnings():
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
            with open(self.outputdirectory+'/stage_2/s2.scousepy', 'wb') as fh:
                pickle.dump((self.saa_dict, self.fitcount), fh)

        # close all figures before moving on
        # (only needed for plt.ion() case)
        plt.close('all')

        return self

    def load_stage_2(self, fn):
        with open(fn, 'rb') as fh:
            self.saa_dict, self.fitcount = pickle.load(fh)
        self.completed_stages.append('s2')

    def stage_3(self, tol, njobs=1, verbose=False, spatial=False,
                clear_cache=True, autosave=True):
        """
        Stage 3

        Automated fitting of the data.

        Parameters
        ----------
        tol : array like
            Tolerance values for the fitting. Should be in the form
            tol = [T1, T2, T3, T4, T4]. See Henshaw et al. 2016a for full
            explanation but in short:
            T1 = multiple of the rms noise value (all components below this
                 value are rejected).
            T2 = minimum width of a component (in channels)
            T3 = Governs how much the velocity of a given component can differ
                 from the closest matching component in the SAA fit. It is
                 given as a multiple of the velocity dispersion of the closest
                 matching component.
            T4 = Similar to T3. Governs how much the velocity dispersion of a
                 given component can differ from the velocity dispersion of the
                 closest matching component in the parent SAA.
            T5 = Dictates how close two components have to be before they are
                 considered indistinguishable. Given as a multiple of the
                 velocity dispersion of the narrowest neighbouring component.
        njobs : int, optional
            Used for parallelised fitting. The parallelisation is a bit crummy
            at the minute - I need to work on this.
        verbose : bool, optional
            Verbose output of the fitting process.
        spatial : bool, optional
            An extra layer of spatial fitting - this isn't implemented yet. Its
            largely covered by the SAA fits but it might be worthwhile
            implementing in the future.
        clear_cache : bool, optional
            Gets rid of the dead weight. Scouse generates *big* output files.
        autosave : bool, optional
            Autosaves the scouse file.

        """

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
            with open(self.outputdirectory+'/stage_3/s3.scousepy', 'wb') as fh:
                pickle.dump((self.indiv_dict, self.tolerances), fh)

        return self

    def load_indiv_dicts(self, fn, stage):
        if stage=='s6':
            with open(fn, 'rb') as fh:
                self.indiv_dict, self.fitcounts6 = pickle.load(fh)
        else:
            with open(fn, 'rb') as fh:
                self.indiv_dict = pickle.load(fh)
        self.completed_stages.append(stage)

    def load_stage_3(self, fn):
        with open(fn, 'rb') as fh:
            self.indiv_dict, self.tolerances = pickle.load(fh)
        self.completed_stages.append('s3')

    def stage_4(self, verbose=False, autosave=True):
        """
        Stage 4

        Select the best fits out of those performed in stage 3.

        Parameters
        ----------
        verbose : bool, optional
            Verbose output.
        autosave : bool, optional
            Autosaves the scouse file.

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
            with open(self.outputdirectory+'/stage_4/s4.scousepy', 'wb') as fh:
                pickle.dump(self.indiv_dict, fh)

        return self

    def load_stage_4(self, fn):
        return self.load_indiv_dicts(fn, stage='s4')

    def stage_5(self, blocksize = 6, plot_residuals=False, figsize=[10,10],
                verbose=False, autosave=True, bitesize=False, repeat=False,
                newfile=None):
        """
        Stage 5

        In this stage the user is required to check the best-fitting solutions

        Parameters
        ----------
        blocksize : int, optional
            Defines the number of spectra that will be checked at any one time.
            Scouse will display blocksize x blocksize spectra.
        plot_residuals : bool, optional
            If true, scouse will display the residuals as well as the best
            fitting solution.
        figsize : list
            Sets the figure size
        verbose : bool, optional
            Verbose output.
        autosave : bool, optional
            Autoaves the scouse output.
        bitesize : bool, optional
            Optional bitesize checking. This allows the user to pick up where
            they left off and continue to check spectra.
        repeat : bool, optional
            Sometimes you may want to run stage 5 multiple times. Combined with
            newfile, this allows you to. If you are repeating the process, set
            to true.
        newfile : string, optional
            If a string scouse will write the output to a new file rather than
            overwriting the previous one. If nothing is entered, but repeat is
            True then scouse will simply append '.bk' to the old s5 output.

        """
        if not bitesize:
            self.check_spec_indices = []
            self.check_block_indices = []

        self.blocksize = blocksize

        s5dir = os.path.join(self.outputdirectory, 'stage_5')
        self.stagedirs.append(s5dir)
        # create the stage_5 directory
        mkdir_s5(self.outputdirectory, s5dir)

        starttime = time.time()

        if verbose:
            progress_bar = print_to_terminal(stage='s5', step='start')

        # Begin interactive plotting
        interactive_state = plt.matplotlib.rcParams['interactive']

        # First create an interactive plot displaying the main diagnostics of
        # 'goodness of fit'. The user can use this to select regions which look
        # bad and from there, select spectra to refit.
        dd = DiagnosticImageFigure(self, blocksize=blocksize, savedir=s5dir,
                                   repeat=repeat, verbose=verbose)
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

        # These are provided by the user during the interactive selection stage
        check_spec_indices = dd.check_spec_indices
        check_block_indices = dd.check_block_indices

        # For staged_checking - check and flatten
        self.check_spec_indices, self.check_block_indices = \
                check_and_flatten(self, check_spec_indices, check_block_indices)
        self.check_spec_indices = np.asarray(self.check_spec_indices)
        self.check_block_indices = np.asarray(self.check_block_indices)

        endtime = time.time()
        if verbose:
            progress_bar = print_to_terminal(stage='s5', step='end', \
                                             t1=starttime, t2=endtime, \
                                             var=[np.size(self.check_spec_indices)+\
                                                 (np.size(self.check_block_indices)*\
                                                 (self.blocksize**2)),
                                                  np.size(self.check_block_indices), \
                                                  np.size(self.check_spec_indices)])

        self.check_spec_indices = np.asarray([index for index in self.check_spec_indices if np.isfinite(index)])
        self.completed_stages.append('s5')

        # Save the scouse object automatically - create a backup if the user
        # wishes to iterate over s5 + s6
        if autosave:
            if repeat:
                if newfile is not None:
                    with open(self.outputdirectory+self.filename+'/stage_5/'+newfile,'wb') as fh:
                        pickle.dump((self.check_spec_indices,
                                     self.check_block_indices,
                                     self.blocksize), fh)
                else:
                    os.rename(self.outputdirectory+self.filename+'/stage_5/s5.scousepy', \
                              self.outputdirectory+self.filename+'/stage_5/s5.scousepy.bk')
                    with open(self.outputdirectory+'/stage_5/s5.scousepy', 'wb') as fh:
                        pickle.dump((self.check_spec_indices,
                                     self.check_block_indices,
                                     self.blocksize), fh)
            else:
                with open(self.outputdirectory+'/stage_5/s5.scousepy', 'wb') as fh:
                    pickle.dump((self.check_spec_indices,
                                 self.check_block_indices,
                                 self.blocksize), fh)

        # close all figures before moving on
        # (only needed for plt.ion() case)
        plt.close('all')

        return self

    def load_stage_5(self, fn):
        with open(fn, 'rb') as fh:
            self.check_spec_indices, self.check_block_indices, \
            self.blocksize = pickle.load(fh)
        self.completed_stages.append('s5')

    def stage_6(self, plot_neighbours=False, radius_pix=1, figsize=[10,10],
                plot_residuals=False, verbose=False, autosave=True,
                blocks_only=False, indiv_only=False, bitesize=False, nspec=None,
                write_ascii=False, repeat=None, newfile=None,
                njobs=1 ):
        """
        Stage 6

        Take a closer look at the spectra selected in s5

        Parameters
        ----------
        plot_neighbours : bool, optional
            Plots the neighbouring pixels before refitting, for context/a
            reminder as to why it was selected in the first place.
        radius_pix : int, optional
            Combined with plot_neighbours - select how many neighbours you want
            to plot.
        figsize : list
            Figure plot size.
        plot_residuals : bool, optional
            If true, scouse will display the residuals as well as the best
            fitting solution.
        verbose : bool, optional
            Verbose output.
        autosave : bool, optional
            Autoaves the scouse output.
        blocks_only : bool, optional
            Allows the user to fit only the blocks.
        indiv_only : bool, optional
            Allows the user to fit only the individual spectra.
        bitesize : bool, optional
            Allows the user to fit the individual spectra in chunks.
        nspec : int, optional
            The number of spectra to be fit during bitesize fitting.
        write_ascii : bool, optional
            Outputs an ascii table containing the best fitting solutions to the
            individual spectra.
        repeat : bool, optional
            Sometimes you may want to run stage 6 multiple times. Combined with
            newfile, this allows you to. If you are repeating the process, set
            to true.
        newfile : bool, optional
            If true, scouse will write the output to a new file rather than
            overwriting the previous one.
        njobs : int, optional
            Used for parallelised fitting. The parallelisation is a bit crummy
            at the minute - I need to work on this.

        """
        # temporary fix: eventually, this should look like stage 2, with
        # interactive figures
        interactive_state = plt.matplotlib.rcParams['interactive']
        plt.ion()

        s6dir = os.path.join(self.outputdirectory, 'stage_6')
        self.stagedirs.append(s6dir)
        # create the stage_6 directory
        mkdir_s6(self.outputdirectory, s6dir)

        starttime = time.time()

        if verbose:
            progress_bar = print_to_terminal(stage='s6', step='start')

        # Firstly check the check_spec_indices against the blocks and remove any
        # duplicates
        self.check_spec_indices = check_blocks(self)
        self.check_spec_indices = np.asarray([index for index in self.check_spec_indices if np.isfinite(index)])

        # Give the user the option of fitting only blocks or individal spectra
        fit_blocks=True; fit_indiv=True
        if blocks_only:
            fit_indiv=False
        elif indiv_only:
            fit_blocks=False

        # bitesize fitting preparation
        if bitesize:
            if self.fitcounts6 != 0.0:
                lower = int(self.fitcounts6)
                upper = int(lower+nspec)
            else:
                lower = 0
                upper = int(lower+nspec)

        # Set ranges for bitesize fitting
        if not bitesize:
            fitrange=np.arange(0,int(np.size(self.check_spec_indices)))
        else:
            if upper>=np.size(self.check_spec_indices):
                if lower >= np.size(self.check_spec_indices):
                    fitrange=[]
                else:
                    fitrange=np.arange(int(lower),\
                                       int(np.size(self.check_spec_indices)))
            else:
                fitrange=np.arange(int(lower),int(upper))

        if fit_indiv:
            # determine how many fits we will actually be performing
            n_to_fit = np.size(fitrange)

            if n_to_fit <= 0:
                raise ValueError(colors.fg._red_+"No spectra are selected to be "+
                                 "fit. Re-fitting individual spectra has "+
                                 "completed."+colors._endc_)

            # Loop through the spectra that are to be fit
            for i_,i in enumerate(fitrange):
                print(colors.fg._lightgrey_+"====================================================="+colors._endc_)
                print(colors.fg._lightgrey_+"Checking {0} out of {1}".format(i_+1, n_to_fit)+colors._endc_)
                print(colors.fg._lightgrey_+"====================================================="+colors._endc_)

                # Here we will fit the individual spectra that have been
                # selected for refitting

                key = self.check_spec_indices[i]

                # This first of all plots the neighbouring pixels - this can be
                # useful if you forget why you selected that spectrum in the
                # first place - it helps to provide a bit of context
                if plot_neighbours:
                    # Find the neighbours
                    indices_adjacent = neighbours(self.cube.shape[1:],
                                                  int(key), radius_pix)
                    # plot the neighbours
                    plot_neighbour_pixels(self, indices_adjacent, figsize)

                # This will plot the current model solution as well as all
                # possible alternatives. The user should either select one of
                # these or press enter to enter the manual fitting mode
                models, selection = plot_alternatives(self, key, figsize, \
                                                  plot_residuals=plot_residuals)
                update_models(self, key, models, selection)

                self.fitcounts6+=1

        if fit_blocks:
            # Stage 5 gives the user the option to select all pixels within a
            # block for refitting - this first of all generates a pseudo-SAA
            # from the block, we then manually fit. This solution is then
            # applied to all spectra contained within the block.
            block_dict={}
            # cycle through all the blocks
            for blocknum in self.check_block_indices:
                # create an empty spectrum
                spec = np.zeros(self.cube.shape[0])
                # get all of the individual pixel indices contained within that
                # block
                block_indices = get_block_indices(self, blocknum)
                # turn the flattened indices into 2D indices such that we can
                # find the spectra in the cube
                coords = gen_2d_coords(self,block_indices)
                # create an SAA
                SAA = gen_pseudo_SAA(self, coords, block_dict, blocknum, spec)
                # prepare the spectra for fitting
                initialise_indiv_spectra_s6(self, SAA, njobs)
                # Manual fitting of the blocks
                manually_fit_blocks(self, block_dict, blocknum)
                self.blockcount+=1
            # automated fitting of block spectra
            auto_fit_blocks(self, block_dict, njobs, self.blocksize, \
                            verbose=verbose)

        if write_ascii and (self.fitcounts6==int(np.size(self.check_spec_indices))) \
           and (self.blockcount == int(np.size(self.check_block_indices))):
            output_ascii_indiv(self, s6dir)

        if (self.fitcounts6 == int(np.size(self.check_spec_indices))) \
           and (self.blockcount == int(np.size(self.check_block_indices))):
            self.fitcounts6 = 0

        endtime = time.time()
        if verbose:
            progress_bar = print_to_terminal(stage='s6', step='end',
                                             t1=starttime, t2=endtime)

        # Save the scouse object automatically - create a backup if the user
        # wishes to iterate over s5 + s6
        if autosave:
            if repeat:
                if newfile is not None:
                    with open(self.outputdirectory+self.filename+'/stage_6/'+newfile, 'wb') as fh:
                        pickle.dump((self.indiv_dict, self.fitcounts6), fh)
                else:
                    os.rename(self.outputdirectory+self.filename+'/stage_6/s6.scousepy', \
                              self.outputdirectory+self.filename+'/stage_6/s6.scousepy.bk')
                    with open(self.outputdirectory+'/stage_6/s6.scousepy', 'wb') as fh:
                        pickle.dump((self.indiv_dict, self.fitcounts6), fh)
            else:
                with open(self.outputdirectory+'/stage_6/s6.scousepy', 'wb') as fh:
                    pickle.dump((self.indiv_dict, self.fitcounts6), fh)

        self.completed_stages.append('s6')

        # reset the interactive state to whatever it was before
        plt.matplotlib.rcParams['interactive'] = interactive_state

        # close all figures before moving on
        # (only needed for plt.ion() case)
        plt.close('all')

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
