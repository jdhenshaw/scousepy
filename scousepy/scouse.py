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

from .stage_3 import *
from .stage_4 import *
#from .stage_5 import interactive_plot, DiagnosticImageFigure
#from .stage_6 import *
#from .io import *
#from .saa_description import saa, add_ids
#from .solution_description import fit

from .colors import *

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

# add Python 2 xrange compatibility, to be removed
# later when we switch to numpy loops
if sys.version_info.major >= 3:
    range = range
    proto=3
else:
    range = xrange
    proto=2

try:
    input = raw_input
except NameError:
    pass

class scouse(object):
    """
    the scouse class

    Attributes
    ==========

    Global attributes - user defined attributes
    -------------------------------------------
    These attributes are set in the config file by the user

    config : string
        Path to the configutation file of scousepy. Must be passed to each stage
    datadirectory : string
        Directory containing the datacube
    filename : string
        Name of the file to be loaded
    outputdirectory : string, optional
        Alternate output directory. Deflault is datadirectory
    fittype : string
        Compatible with pyspeckit's models for fitting different types of
        models. Defualt is Gaussian fitting.
    verbose : bool
        Verbose output to terminal
    autosave : bool
        Save the output at each stage of the process.

    Global attributes - scouse defined attributes
    ---------------------------------------------
    cube : spectral cube object
        A spectral cube object generated from the FITS input
    completed_stages : list
        a list to be updated as each stage is completed

    stage 1 - user defined attributes
    ---------------------------------
    write_moments : bool, optional
        If true, scouse will write fits files of the moment maps
    save_fig : bool, optional
        If true, scouse will output a figure of the coverage
    coverage_config_file_path : string
        File path for coverage configuration file
    nrefine : number
        Number of refinement steps - scouse will refine the coverage map
        according to spectral complexity if several wsaas are provided.
    mask_below : number
        Masking value for moment generation. This will also determine which
        pixels scouse fits
    x_range : list
        Data x range in pixels
    y_range : list
        Data y range in pixels
    vel_range : list
        Data velocity range in map units
    wsaa : list
        Width of the spectral averaging areas. The user can provide a list of
        values if they would like to refine the coverage in complex regions.
    fillfactor : list
        Fractional limit below which SAAs are rejected. Again, can be given as
        a list so the user can control which SAAs are selected for fitting.
    samplesize : number
        Sample size for randomly selecting SAAs.
    covmethod : string
        Method used to define the coverage. Choices are 'regular' for normal
        scouse fitting or 'random'. The latter generates a random sample of
        'samplesize' SAAs.
    spacing : string
        Method setting spacing of SAAs. Choices are 'nyquist' or 'regular'.
    speccomplexity : string
        Method defining spectral complexity. Choices include:
        - 'momdiff': which measures the difference between the velocity at
                     peak emission and the
                     first moment.
        - 'kurtosis': bases the spectral complexity on the kurtosis of the
                      spectrum.
    totalsaas : number
        Total number of SAAs.
    totalspec : number
        Total number of spectra within the coverage.

    stage 1 - scouse defined attributes
    -----------------------------------
    lenspec : number
        This will refer to the total number of spectra that scouse will actually
        fit. This includes some spectra being fit multiple times due to overlap
        of SAAs if spacing: 'nyquist' is selected.
    saa_dict : dictionary
        A dictionary containing all of the SAA spectra in scouse format
    rms_approx : number
        An estimate of the mean rms across the map
    x : ndarray
        The spectral axis
    xtrim : ndarray
        The spectral axis trimmed according to vel_range (see above)
    trimids : ndarray
        A mask of x according to vel_range

    """

    def __init__(self, config=''):

        # global -- user
        self.config=config
        self.datadirectory=None
        self.filename=None
        self.outputdirectory=None
        self.fittype=None
        self.verbose=None
        self.autosave=None
        # global -- scousepy
        self.cube=None
        self.completed_stages = []

        # stage 1 -- user
        self.write_moments=None
        self.save_fig=None
        # stage 1 -- scousepy coverage
        self.coverage_config_file_path=None
        self.nrefine=None
        self.mask_below=None
        self.x_range=None
        self.y_range=None
        self.vel_range=None
        self.wsaa=None
        self.fillfactor=None
        self.samplesize=None
        self.covmethod=None
        self.spacing=None
        self.speccomplexity=None
        self.totalsaas=None
        self.totalspec=None
        # stage 1 -- scousepy SAAs
        self.lenspec=None
        self.saa_dict=None
        self.rms_approx=None
        self.x=None
        self.xtrim=None
        self.trimids=None

        # stage 2 -- user
        self.write_ascii=None
        # stage 2 -- scousepy
        self.fitcount=None
        self.modelstore={}

        # stage 3 -- user
        self.tol=None
        self.njobs=None

        # self.stagedirs = []
        # self.cube = None
        # self.config_file=None
        # self.tolerances = None
        # self.specres = None
        # self.fittype = fittype
        # self.sample = None
        # self.x = None
        # self.xtrim = None
        # self.trimids=None
        # self.saa_dict = None
        # self.indiv_dict = None
        # self.key_set = None
        # self.fitcount = None
        #
        # self.fitcounts6 = 0
        # self.blockcount = 0
        # self.blocksize = None
        # self.check_spec_indices = None
        # self.check_block_indices = None
        #

    @staticmethod
    def stage_1(config=''):
        """
        Identify the spatial area over which the fitting will be implemented.

        Parameters
        ----------
        config : string
            Path to the configuration file. This must be provided.

        Notes
        -----
        See scouse class documentation for description of the parameters that
        are set during this stage.

        """

        # Import
        from .stage_1 import generate_SAAs, plot_coverage, compute_noise, get_x_axis
        from .io import import_from_config
        from .verbose_output import print_to_terminal
        from scousepy.scousecoverage import ScouseCoverage

        # Check input
        if os.path.exists(config):
            self=scouse(config=config)
            stages=['stage_1']
            for stage in stages:
                import_from_config(self, config, config_key=stage)
        else:
            print('')
            print(colors.fg._lightred_+"Please supply a valid scousepy configuration file. \n\nEither: \n"+
                                  "1: Check the path and re-run. \n"+
                                  "2: Create a configuration file using 'run_setup'."+colors._endc_)
            print('')
            return

        # check if stage 1 has already been run
        if os.path.exists(self.outputdirectory+self.filename+'/stage_1/s1.scousepy'):
            if self.verbose:
                progress_bar = print_to_terminal(stage='s1', step='load')
            self.load_stage_1(self.outputdirectory+self.filename+'/stage_1/s1.scousepy')
            if 's1' in self.completed_stages:
                print(colors.fg._lightgreen_+"Coverage complete and SAAs initialised. "+colors._endc_)
                print('')
            return self

        # load the cube
        fitsfile = os.path.join(self.datadirectory, self.filename+'.fits')
        self.load_cube(fitsfile=fitsfile)

        #----------------------------------------------------------------------#
        # Main routine
        #----------------------------------------------------------------------#
        if self.verbose:
            progress_bar = print_to_terminal(stage='s1', step='start')

        # Define the coverage
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_log = log.level
            log.setLevel('ERROR')
            # Interactive coverage generator
            coverageobject=ScouseCoverage(scouseobject=self,verbose=self.verbose)
            coverageobject.show()
            # write out the config file for the coverage
            self.coverage_config_file_path=os.path.join(self.outputdirectory,self.filename,'config_files','coverage.config')
            with open(self.coverage_config_file_path, 'w') as file:
                for line in coverageobject.config_file:
                    file.write(line)
            # set the parameters
            import_from_config(self, self.coverage_config_file_path)
            log.setLevel(old_log)

        # start the time once the coverage has been generated
        starttime = time.time()

        # Create a dictionary to store the SAAs
        self.saa_dict = {}
        # Compute typical noise within the spectra
        self.rms_approx = compute_noise(self)
        # Generate the x axis common to the fitting process
        self.x, self.xtrim, self.trimids = get_x_axis(self)

        # Generate the SAAs
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_log = log.level
            log.setLevel('ERROR')
            generate_SAAs(self, coverageobject, verbose=self.verbose)
            log.setLevel(old_log)

        # Saving figures
        if self.save_fig:
            coverage_plot_filename=os.path.join(self.outputdirectory,self.filename,'stage_1','coverage.pdf')
            plot_coverage(self, coverageobject, coverage_plot_filename)

        # Write moments as fits files
        if self.write_moments:
            momentoutputdir=os.path.join(self.outputdirectory,self.filename,'stage_1/')
            from .io import output_moments
            output_moments(self.cube.header,coverageobject.moments,momentoutputdir,self.filename)

        # Wrapping up
        plt.close('all')
        endtime = time.time()
        if self.verbose:
            progress_bar = print_to_terminal(stage='s1', step='end',
                                             length=np.size(coverageobject.moments[0].value),
                                             t1=starttime, t2=endtime)
        self.completed_stages.append('s1')

        # Save the scouse object automatically
        if self.autosave:
            import pickle
            with open(self.outputdirectory+self.filename+'/stage_1/s1.scousepy', 'wb') as fh:
                pickle.dump((self.completed_stages,
                             self.coverage_config_file_path,
                             self.lenspec,
                             self.saa_dict,
                             self.x,
                             self.xtrim,
                             self.trimids), fh, protocol=proto)

        return self

    def load_stage_1(self,fn):
        """
        Method used to load in the progress of stage 1
        """
        import pickle
        with open(fn, 'rb') as fh:
            self.completed_stages,\
            self.coverage_config_file_path,\
            self.lenspec,\
            self.saa_dict,\
            self.x,\
            self.xtrim,\
            self.trimids=pickle.load(fh)

    def stage_2(config=''):
        """
        Fitting of the SAAs

        Parameters
        ----------
        config : string
            Path to the configuration file. This must be provided.

        Notes
        -----
        See scouse class documentation for description of the parameters that
        are set during this stage.

        """
        # import
        from .io import import_from_config
        from .verbose_output import print_to_terminal
        from scousepy.scousefitter import ScouseFitter
        from .model_housing2 import saamodel
        from .stage_2 import generate_saa_list

        # Check input
        if os.path.exists(config):
            self=scouse(config=config)
            stages=['stage_1','stage_2']
            for stage in stages:
                import_from_config(self, config, config_key=stage)
        else:
            print('')
            print(colors.fg._lightred_+"Please supply a valid scousepy configuration file. \n\nEither: \n"+
                                  "1: Check the path and re-run. \n"+
                                  "2: Create a configuration file using 'run_setup'."+colors._endc_)
            print('')
            return

        # check if stages 1 and 2 have already been run
        if os.path.exists(self.outputdirectory+self.filename+'/stage_1/s1.scousepy'):
            self.load_stage_1(self.outputdirectory+self.filename+'/stage_1/s1.scousepy')
            import_from_config(self, self.coverage_config_file_path)
        if os.path.exists(self.outputdirectory+self.filename+'/stage_2/s2.scousepy'):
            if self.verbose:
                progress_bar = print_to_terminal(stage='s2', step='load')
            self.load_stage_2(self.outputdirectory+self.filename+'/stage_2/s2.scousepy')
            if self.fitcount is not None:
                if np.all(self.fitcount):
                    print(colors.fg._lightgreen_+"All spectra have solutions. Fitting complete. "+colors._endc_)
                    print('')
                    return self

        # load the cube
        fitsfile = os.path.join(self.datadirectory, self.filename+'.fits')
        self.load_cube(fitsfile=fitsfile)

        #----------------------------------------------------------------------#
        # Main routine
        #----------------------------------------------------------------------#
        if self.verbose:
            progress_bar = print_to_terminal(stage='s2', step='start')

        # generate a list of all SAA's (inc. all wsaas)
        saa_list = generate_saa_list(self)
        saa_list = np.asarray(saa_list)

        # Record which spectra have been fit - first check to see if this has
        # already been created
        if self.fitcount is None:
            # if it hasn't
            self.fitcount=np.zeros(int(np.sum(self.totalsaas)), dtype='bool')

        starttime = time.time()

        fitterobject=ScouseFitter(self.modelstore, method='scouse',
                                spectra=saa_list[:,0],
                                scouseobject=self,
                                SAA_dict=self.saa_dict,
                                parent=saa_list[:,1],
                                fitcount=self.fitcount,
                                )
        fitterobject.show()

        # Now we want to go through and add the model solutions to the SAAs
        for key in range(len(saa_list[:,0])):
            # identify the right dictionary
            saa_dict=self.saa_dict[saa_list[key,1]]
            # retrieve the SAA
            SAA=saa_dict[saa_list[key,0]]
            # obtain the correct model from modelstore
            modeldict=self.modelstore[key]
            # convert the modelstore dictionary into an saamodel object
            model=saamodel(modeldict)
            # add this to the SAA
            SAA.add_saamodel(model)

        # Wrapping up
        endtime = time.time()
        if self.verbose:
            progress_bar = print_to_terminal(stage='s2', step='end',
                                             t1=starttime, t2=endtime)

        # Check that the fitting has completed
        if np.all(self.fitcount):
            if self.write_ascii:
                from .io import output_ascii_saa
                output_ascii_saa(self, os.path.join(self.outputdirectory,self.filename,'stage_2/'))
            self.completed_stages.append('s2')

        # Save the scouse object automatically
        if self.autosave:
            import pickle
            with open(self.outputdirectory+self.filename+'/stage_2/s2.scousepy', 'wb') as fh:
                pickle.dump((self.completed_stages,
                            self.saa_dict,
                            self.fitcount,
                            self.modelstore), fh, protocol=proto)

        return self

    def load_stage_2(self, fn):
        import pickle
        with open(fn, 'rb') as fh:
            self.completed_stages,\
            self.saa_dict, \
            self.fitcount, \
            self.modelstore = pickle.load(fh)

    def stage_3(config=''):
        """
        Stage 3

        Automated fitting of the data.

        Parameters
        ----------
        config : string
            Path to the configuration file. This must be provided.

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
        # import
        from .io import import_from_config
        from .verbose_output import print_to_terminal

        # Check input
        if os.path.exists(config):
            self=scouse(config=config)
            stages=['stage_1','stage_2','stage_3']
            for stage in stages:
                import_from_config(self, config, config_key=stage)
        else:
            print('')
            print(colors.fg._lightred_+"Please supply a valid scousepy configuration file. \n\nEither: \n"+
                                  "1: Check the path and re-run. \n"+
                                  "2: Create a configuration file using 'run_setup'."+colors._endc_)
            print('')
            return

        # check if stages 1, 2, and 3 have already been run
        if os.path.exists(self.outputdirectory+self.filename+'/stage_1/s1.scousepy'):
            self.load_stage_1(self.outputdirectory+self.filename+'/stage_1/s1.scousepy')
            import_from_config(self, self.coverage_config_file_path)
        if os.path.exists(self.outputdirectory+self.filename+'/stage_2/s2.scousepy'):
            self.load_stage_2(self.outputdirectory+self.filename+'/stage_2/s2.scousepy')
            if self.fitcount is not None:
                if not np.all(self.fitcount):
                    print(colors.fg._lightred_+"Not all spectra have solutions. Please complete stage 2 before proceding. "+colors._endc_)
        if os.path.exists(self.outputdirectory+self.filename+'/stage_3/s3.scousepy'):
            if self.verbose:
                progress_bar = print_to_terminal(stage='s3', step='load')
            self.load_stage_3(self.outputdirectory+self.filename+'/stage_3/s3.scousepy')
            if 's3' in self.completed_stages:
                print(colors.fg._lightgreen_+"Fitting completed. "+colors._endc_)
                print('')
            return self

        # load the cube
        fitsfile = os.path.join(self.datadirectory, self.filename+'.fits')
        self.load_cube(fitsfile=fitsfile)

        #----------------------------------------------------------------------#
        # Main routine
        #----------------------------------------------------------------------#
        if self.verbose:
            progress_bar = print_to_terminal(stage='s3', step='start')

        starttime = time.time()
        # create a list that is going to house all instances of the
        # individual_spectrum class. We want this to be a list for ease of
        # parallelisation.
        starttimeinit=time.time()
        indivspec_list=initialise_fitting(self)
        endtimeinit=time.time()
        if self.verbose:
            progress_bar = print_to_terminal(stage='s3', step='initend',t1=starttimeinit, t2=endtimeinit)

        # determine cpus to use
        if self.njobs==None:
            self.get_njobs()

        # now begin the fitting
        starttimefitting=time.time()
        indivspec_list_completed=autonomous_decomposition(self, indivspec_list)
        endtimefitting=time.time()
        if self.verbose:
            progress_bar = print_to_terminal(stage='s3', step='fitend',
                                        t1=starttimefitting, t2=endtimefitting)

        # now compile the spectra
        starttimecompile=time.time()
        compile_spectra(self, indivspec_list_completed)
        endtimecompile=time.time()
        if self.verbose:
            progress_bar = print_to_terminal(stage='s3', step='compileend',
                                        t1=starttimecompile, t2=endtimecompile)

        # Wrapping up
        endtime = time.time()
        if self.verbose:
            progress_bar = print_to_terminal(stage='s3', step='end',
                                             t1=starttime, t2=endtime)
        self.completed_stages.append('s3')

        # Save the scouse object automatically
        if self.autosave:
            import pickle
            with open(self.outputdirectory+self.filename+'/stage_3/s3.scousepy', 'wb') as fh:
                pickle.dump((self.completed_stages, self.indiv_dict), fh, protocol=proto)

        return self

    def load_stage_3(self, fn):
        import pickle
        with open(fn, 'rb') as fh:
            self.completed_stages,\
            self.indiv_dict = pickle.load(fh)

    def load_indiv_dicts(self, fn, stage):
        if stage=='s6':
            with open(fn, 'rb') as fh:
                self.indiv_dict, self.fitcounts6 = pickle.load(fh)
        else:
            with open(fn, 'rb') as fh:
                self.indiv_dict = pickle.load(fh)

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

    def run_setup(filename, datadirectory, outputdir=None,
                  config_filename='scousepy.config', description=True,
                  verbose=True):
        """
        Generates a scousepy configuration file

        Parameters
        ----------
        filename : string
            Name of the file to be loaded
        datadirectory : string
            Directory containing the datacube
        outputdir : string, optional
            Alternate output directory. Deflault is datadirectory
        config_filename : string, optional
            output filename for the configuration file
        description : bool, optional
            whether or not a description of each parameter is included in the
            configuration file
        verbose : bool, optional
            verbose output to terminal

        """
        from .io import create_directory_structure
        from .io import generate_config_file
        from .verbose_output import print_to_terminal

        if outputdir is None:
            outputdir=datadirectory

        scousedir=os.path.join(outputdir, filename)
        configdir=os.path.join(scousedir+'/config_files')
        configpath=os.path.join(scousedir+'/config_files', config_filename)

        if verbose:
            progress_bar = print_to_terminal(stage='init', step='init')

        if os.path.exists(configpath):
            if verbose:
                progress_bar = print_to_terminal(stage='init', step='configexists')
            return os.path.join(scousedir+'/config_files', config_filename)
        else:
            if not os.path.exists(scousedir):
                create_directory_structure(scousedir)
                generate_config_file(filename, datadirectory, outputdir, configdir, config_filename, description)
                if verbose:
                    progress_bar = print_to_terminal(stage='init', step='makingconfig')
            else:
                configpath=None
                print('')
                print(colors.fg._yellow_+"Warning: output directory exists but does not contain a config file. "+colors._endc_)
                print('')

        return configpath

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
        import warnings

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
            log.setLevel(old_log)

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
# Methods
#==============================================================================#

    def get_njobs(self):
        """
        Determines number of cpus available for parallel processing. If njobs
        is set to None scouse will automatically use 75% of available cpus.

        """
        import multiprocessing
        maxcpus=multiprocessing.cpu_count()
        self.njobs=int(0.75*maxcpus)

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
