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

from .stage_2 import *
from .stage_3 import *
from .stage_4 import *
from .stage_5 import interactive_plot, DiagnosticImageFigure
from .stage_6 import *
from .io import *
#from .saa_description import saa, add_ids
#from .solution_description import fit
from .model_housing import *
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
        self.config_table=None
        self.tolerances = None
        self.specres = None
        self.fittype = fittype
        self.sample = None
        self.x = None
        self.xtrim = None
        self.trimids=None
        self.saa_dict = None
        self.indiv_dict = None
        self.key_set = None
        self.fitcount = None
        self.modelstore = {}
        self.fitcounts6 = 0
        self.blockcount = 0
        self.blocksize = None
        self.check_spec_indices = None
        self.check_block_indices = None
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

            self.cube = _cube

    @staticmethod
    def stage_1(filename, datadirectory, outputdir=None, fittype='gaussian',
                write_moments=False, save_fig=False, autosave=True,
                verbose = False ):
        """
        Stage 1

        Identify the spatial area over which the fitting will be implemented.

        Parameters
        ----------
        filename : string
            Name of the file to be loaded
        datadirectory : string
            Directory containing the datacube
        outputdir : string, optional
            Alternate output directory. Deflault is datadirectory
        fittype : string
            Compatible with pyspeckit's models for fitting different types of
            models. Defualt is Gaussian fitting.
        write_moments : bool, optional
            If true, scouse will write fits files of the moment maps
        save_fig : bool, optional
            If true, scouse will output a figure of the coverage
        autosave : bool, optional
            Save the output at each stage of the process.
        verbose : bool, optional
            Verbose output to terminal
        """

        # import
        from .stage_1 import generate_SAAs, plot_coverage, compute_noise, get_x_axis

        if outputdir is None:
            outputdir=datadirectory

        # set the basics
        self = scouse(fittype=fittype, filename=filename, outputdir=outputdir,
                      datadirectory=datadirectory)

        # fits file name
        fitsfile = os.path.join(datadirectory, self.filename+'.fits')

        # load the cube
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_log = log.level
            log.setLevel('ERROR')

            self.load_cube(fitsfile=fitsfile)

            log.setLevel(old_log)

        # check to see if output files already exist
        if os.path.exists(outputdir+filename+'/stage_1/s1.scousepy'):
            if verbose:
                progress_bar = print_to_terminal(stage='s1', step='load')
            self.load_stage_1(outputdir+filename+'/stage_1/s1.scousepy')
            return self

        # directory structure
        s1dir = os.path.join(outputdir, self.filename, 'stage_1')
        self.stagedirs.append(s1dir)
        # create the stage_1 directory
        mkdir_s1(self.outputdirectory, s1dir)

        # verbose output
        if verbose:
            progress_bar = print_to_terminal(stage='s1', step='start')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_log = log.level
            log.setLevel('ERROR')

            # Interactive coverage generator
            from scousepy.scousecoverage import ScouseCoverage
            coverageobject=ScouseCoverage(scouseobject=self, verbose=verbose)
            coverageobject.show()
            self.config_table=coverageobject.config_table

            log.setLevel(old_log)

        # Main routine
        starttime = time.time()

        # create a dictionary to store the SAAs
        self.saa_dict = {}

        # Compute typical noise within the spectra
        self.rms_approx = compute_noise(self)
        # Generate the x axis common to the fitting process
        self.x, self.xtrim, self.trimids = get_x_axis(self, coverageobject)

        # Generate the SAAs
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_log = log.level
            log.setLevel('ERROR')

            generate_SAAs(self, coverageobject, verbose=verbose)

            log.setLevel(old_log)

        # Saving figures
        if save_fig:
            plot_coverage(self, coverageobject, s1dir, filename)

        # Write moments as fits files
        if write_moments:
            from .io import output_moments
            output_moments(self.cube.header,coverageobject.moments,s1dir,filename)

        # Wrapping up
        endtime = time.time()
        if verbose:
            progress_bar = print_to_terminal(stage='s1', step='end',
                                             length=np.size(coverageobject.moments[0].value),
                                             t1=starttime, t2=endtime)

        self.completed_stages.append('s1')

        # Save the scouse object automatically
        if autosave:
            with open(self.outputdirectory+'/stage_1/s1.scousepy', 'wb') as fh:
                pickle.dump((self.datadirectory,
                             self.filename,
                             self.outputdirectory,
                             self.config_table,
                             self.completed_stages,
                             self.saa_dict,
                             self.x,
                             self.xtrim,
                             self.trimids), fh)
        return self

    def load_stage_1(self,fn):
        """
        Method used to load in the progress of stage 1
        """
        with open(fn, 'rb') as fh:
            self.datadirectory,\
            self.filename,\
            self.outputdirectory,\
            self.config_table,\
            self.completed_stages,\
            self.saa_dict,\
            self.x,\
            self.xtrim,\
            self.trimids=pickle.load(fh)

    def stage_2(self, verbose = False, write_ascii=False, autosave=True,
                bitesize=False, nspec=None, training_set=False, derivspec=False):
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

        # Total number of spectra to be fit across all SAAs
        speccount=np.size(saa_list[:,0])
        spectratobefit=saa_list[:,0]
        parent=saa_list[:,1]

        # Record which spectra have been fit - first check to see if this has
        # already been created
        if self.fitcount is None:
            # if it hasn't
            self.fitcount=np.zeros(speccount, dtype='bool')

        # load in the fitter
        from scousepy.scousefitter import ScouseFitter

        if verbose:
            progress_bar = print_to_terminal(stage='s2', step='start')

        starttime = time.time()

        myfitter=ScouseFitter(self.modelstore, method='scouse',
                              spectra=spectratobefit,
                              scouseobject=self,
                              SAA_dict=self.saa_dict,
                              parent=parent,
                              fitcount=self.fitcount,
                              )
        myfitter.show()

        endtime = time.time()
        if verbose:
            progress_bar = print_to_terminal(stage='s2', step='end',
                                             t1=starttime, t2=endtime)


        # loop through the saa sizes
        # for i in range(len(self.wsaa)):
        #     saa_dict=self.saa_dict[i]
        #     spectratobefit=spectra_to_be_fit(self,saa_dict)
        #
        #     sys.exit()
        #
        # sys.exit()
        #
        #
        #
        # # bitesize fitting preparation
        # if bitesize:
        #     if self.fitcount != 0.0:
        #         lower = int(self.fitcount)
        #         upper = int(lower+nspec)
        #     else:
        #         lower = 0
        #         upper = int(lower+nspec)
        #

        #
        # # Set ranges for bitesize fitting
        # if not bitesize:
        #     fitrange=np.arange(0,int(np.size(saa_list[:,0])))
        # else:
        #     if upper>=np.size(saa_list[:,0]):
        #         if lower >= np.size(saa_list[:,0]):
        #             fitrange=[]
        #         else:
        #             fitrange=np.arange(int(lower),int(np.size(saa_list[:,0])))
        #     else:
        #         fitrange=np.arange(int(lower),int(upper))
        #
        # # determine how many fits we will actually be performing
        # n_to_fit = sum([self.saa_dict[saa_list[ii,1]][saa_list[ii,0]].to_be_fit
        #                 for ii in fitrange])
        #
        # if n_to_fit <= 0:
        #     raise ValueError(colors.fg._red_+"No spectra are selected to be fit."+
        #                      "Fitting has completed."+colors._endc_)
        #
        # # Loop through the SAAs
        # for i_,i in enumerate(fitrange):
        #     print(colors.fg._lightgrey_+"====================================================="+colors._endc_)
        #     print(colors.fg._lightgrey_+"Fitting {0} out of {1}".format(i_+1, n_to_fit)+colors._endc_)
        #     print(colors.fg._lightgrey_+"====================================================="+colors._endc_)
        #     # Get the relevant SAA dictionary (if multiple wsaa values are
        #     # supplied)
        #     saa_dict = self.saa_dict[saa_list[i,1]]
        #     # Get the first SAA to fit
        #     SAA = saa_dict[saa_list[i,0]]
        #
        #     # Fitting process is different for the first SAA in a wsaa loop.
        #     # For all subsequent SAAs scouse will try and apply the previous
        #     # solution to the spectrum in an attempt to speed things up and
        #     # reduce the interactivity
        #     if SAA.index == 0.0:
        #         SAAid=0
        #         firstfit=True
        #     elif i == np.min(fitrange):
        #         SAAid=SAA.index
        #         firstfit=True
        #     elif training_set:
        #         SAAid=SAA.index
        #         firstfit=True
        #
        #     if SAA.to_be_fit:
        #         with warnings.catch_warnings():
        #             warnings.simplefilter('ignore', category=DeprecationWarning)
        #
        #             # enter the fitting process
        #             bf = fitting(self, SAA, saa_dict, SAAid,
        #                          training_set=self.training_set,
        #                          init_guess=firstfit, derivspec=derivspec)
        #         SAAid = SAA.index
        #         firstfit=False
        #
        #     self.fitcount+=1
        #
        # # Output at the end of SAA fitting
        # if write_ascii and (self.fitcount == np.size(saa_list[:,0])):
        #     output_ascii_saa(self, s2dir)
        #     self.completed_stages.append('s2')
        #


        # Save the scouse object automatically
        if autosave:
            with open(self.outputdirectory+'/stage_2/s2.scousepy', 'wb') as fh:
                pickle.dump((self.saa_dict, self.fitcount, self.modelstore), fh)


        # close all figures before moving on
        # (only needed for plt.ion() case)
        plt.close('all')

        return self

    def load_stage_2(self, fn):
        with open(fn, 'rb') as fh:
            self.saa_dict, self.fitcount, self.modelstore = pickle.load(fh)
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

        self.completed_stages.append('s5')

        # Save the scouse object automatically - create a backup if the user
        # wishes to iterate over s5 + s6
        if autosave:
            if repeat:
                if newfile is not None:
                    with open(self.outputdirectory+'/stage_5/'+newfile,'wb') as fh:
                        pickle.dump((self.check_spec_indices,
                                     self.check_block_indices,
                                     self.blocksize), fh)
                else:
                    os.rename(self.outputdirectory+'/stage_5/s5.scousepy', \
                              self.outputdirectory+'/stage_5/s5.scousepy.bk')
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
                    with open(self.outputdirectory+'/stage_6/'+newfile, 'wb') as fh:
                        pickle.dump((self.indiv_dict, self.fitcounts6), fh)
                else:
                    os.rename(self.outputdirectory+'/stage_6/s6.scousepy', \
                              self.outputdirectory+'/stage_6/s6.scousepy.bk')
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
