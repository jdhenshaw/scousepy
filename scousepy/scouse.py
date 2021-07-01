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

from .colors import *

import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt

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

    stage 2 - user defined attributes
    ---------------------------------
    write_ascii : bool
        If True will create an ascii file containing the best-fitting solutions
        to each of the spectral averaging areas.

    stage 2 - scouse defined attributes
    -----------------------------------
    saa_dict : dictionary
        This dictionary houses each of the spectral averaging areas fit by
        scouse
    fitcount : number
        A number indicating how many of the spectra have currently been fit.
        Used so that scouse can remember where it got upto in the fitting
        process
    modelstore : dictionary
        Modelstore contains all the best-fitting solutions while they are
        waiting to be added to saa_dict

    stage 3 - user defined attributes
    ---------------------------------
    tol : list
        Tolerance values for the fitting. Should be in the form
        tol = [T0, T1, T2, T3, T4, T4]. See Henshaw et al. 2016a for full
        explanation but in short:
        T0 = NEW! controls how different the number of components in the fitted
             spectrum can be from the number of components of the parent
             spectrum.

             if |ncomps_spec - ncomps_saa| > T0 ; the fit is rejected

        T1 = multiple of the rms noise value (all components below this
             value are rejected).

             if I_peak < T1*rms ; the component is rejected

        T2 = minimum width of a component (in channels)

             if FHWM < T2*channel_width ; the component is rejected

        T3 = Governs how much the velocity dispersion of a given component can
             differ from the closest matching component in the SAA fit. It is
             given as a multiple of the velocity dispersion of the closest
             matching component.

             relchange = sigma/sigma_saa
             if relchange < 1:
                 relchange = 1/relchange
             if relchange > T3 ; the component is rejected

        T4 = Similar to T3. Governs how much the velocity of a given component
             can differ from the velocity of the closest matching component in
             the parent SAA.

             lowerlim = vel_saa - T4*disp_saa
             upperlim = vel_saa + T4*disp_saa
             if vel < lowerlim or vel > upperlim ; the component is rejected

        T5 = Dictates how close two components have to be before they are
             considered indistinguishable. Given as a multiple of the
             velocity dispersion of the narrowest neighbouring component.

             if vel - vel_neighbour < T5*FWHM_narrowestcomponent ; take the
             average of the two components and use this as a new guess

    njobs : int, optional
        Used for parallelised fitting

    stage 3 - scouse defined attributes
    -----------------------------------
    indiv_dict : dictionary
        A dictionary containing each spectrum fit by scouse and their best
        fitting model solutions


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

        # stage 5 -- scousepy
        self.check_spec_indices=[]

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
    def stage_1(config='', interactive=True):
        """
        Identify the spatial area over which the fitting will be implemented.

        Parameters
        ----------
        config : string
            Path to the configuration file. This must be provided.
        interactive : bool, optional
            Default is to run coverage with interactive GUI, but this can be
            bypassed in favour of using the config file

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
            coverageobject=ScouseCoverage(scouseobject=self,verbose=self.verbose,
                                            interactive=interactive)
            if interactive:
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
            generate_SAAs(self, coverageobject)
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
                             self.trimids,
                             self.rms_approx), fh, protocol=proto)

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
            self.trimids,\
            self.rms_approx=pickle.load(fh)

    def stage_2(config='', refit=False):
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
        from .model_housing import saamodel
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
            #### TMP FIX
            self.coverage_config_file_path=os.path.join(self.outputdirectory,self.filename,'config_files','coverage.config')
            ###
            import_from_config(self, self.coverage_config_file_path)
        if os.path.exists(self.outputdirectory+self.filename+'/stage_2/s2.scousepy'):
            if self.verbose:
                progress_bar = print_to_terminal(stage='s2', step='load')
            self.load_stage_2(self.outputdirectory+self.filename+'/stage_2/s2.scousepy')
            if self.fitcount is not None:
                if np.all(self.fitcount):
                    if not refit:
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

        # generate a list of all SAAs (inc. all wsaas)
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
                                fit_dict=self.saa_dict,
                                parent=saa_list[:,1],
                                fitcount=self.fitcount,
                                refit=refit)
        fitterobject.show()

        if np.all(self.fitcount):
            # Now we want to go through and add the model solutions to the SAAs
            for key in range(len(saa_list[:,0])):
                # identify the right dictionary
                saa_dict=self.saa_dict[saa_list[key,1]]
                # retrieve the SAA
                SAA=saa_dict[saa_list[key,0]]
                # obtain the correct model from modelstore
                modeldict=self.modelstore[key]
                # if there are any zero component fits then mark these as
                # SAA.to_be_fit==False
                if not modeldict['fitconverge']:
                    setattr(SAA, 'to_be_fit', False)
                else:
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
                    return
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

        # model selection
        starttimemodelselection = time.time()
        if self.verbose:
            progress_bar = print_to_terminal(stage='s3', step='modelselectstart')
        # select the best model out of those available - i.e. that with the
        # lowest aic value
        model_selection(self)
        endtimemodelselection = time.time()
        if self.verbose:
            progress_bar = print_to_terminal(stage='s3', step='modelselectend',
                        t1=starttimemodelselection, t2=endtimemodelselection)

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

    def stage_4(config='', bitesize=False, verbose=True):
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
        # import
        from .io import import_from_config
        from .verbose_output import print_to_terminal
        from scousepy.scousefitchecker import ScouseFitChecker

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

        # check if stages 1, 2, 3 and 4 have already been run
        if os.path.exists(self.outputdirectory+self.filename+'/stage_1/s1.scousepy'):
            self.load_stage_1(self.outputdirectory+self.filename+'/stage_1/s1.scousepy')
            #### TMP FIX
            self.coverage_config_file_path=os.path.join(self.outputdirectory,self.filename,'config_files','coverage.config')
            ###
            import_from_config(self, self.coverage_config_file_path)
        if os.path.exists(self.outputdirectory+self.filename+'/stage_2/s2.scousepy'):
            self.load_stage_2(self.outputdirectory+self.filename+'/stage_2/s2.scousepy')
            if self.fitcount is not None:
                if not np.all(self.fitcount):
                    print(colors.fg._lightred_+"Not all spectra have solutions. Please complete stage 2 before proceding. "+colors._endc_)
                    return
        if os.path.exists(self.outputdirectory+self.filename+'/stage_3/s3.scousepy'):
            self.load_stage_3(self.outputdirectory+self.filename+'/stage_3/s3.scousepy')
        if os.path.exists(self.outputdirectory+self.filename+'/stage_4/s4.scousepy'):
            if self.verbose:
                progress_bar = print_to_terminal(stage='s4', step='load')
            self.load_stage_4(self.outputdirectory+self.filename+'/stage_4/s4.scousepy')
            if not bitesize:
                print(colors.fg._lightgreen_+"Fit check already complete. Use bitesize=True to re-enter model checker. "+colors._endc_)
                print('')
                return self

        # load the cube
        fitsfile = os.path.join(self.datadirectory, self.filename+'.fits')
        self.load_cube(fitsfile=fitsfile)

        #----------------------------------------------------------------------#
        # Main routine
        #----------------------------------------------------------------------#

        starttime = time.time()

        if np.size(self.check_spec_indices)==0:
            if verbose:
                 progress_bar = print_to_terminal(stage='s4', step='start')

        # Interactive coverage generator
        fitcheckerobject=ScouseFitChecker(scouseobject=self, selected_spectra=self.check_spec_indices)
        fitcheckerobject.show()

        if bitesize:
            self.check_spec_indices=self.check_spec_indices+fitcheckerobject.check_spec_indices
            self.check_spec_indices=list(set(self.check_spec_indices))
        else:
            self.check_spec_indices=fitcheckerobject.check_spec_indices

        # for key in self.indiv_dict.keys():
        #     print(key, self.indiv_dict[key])

        # print('')

        sorteddict={}
        for sortedkey in sorted(self.indiv_dict.keys()):
            sorteddict[sortedkey]=self.indiv_dict[sortedkey]

        self.indiv_dict=sorteddict

        # for key in self.indiv_dict.keys():
        #     print(key, self.indiv_dict[key])

        # print('')

        # Wrapping up
        endtime = time.time()
        if self.verbose:
            progress_bar = print_to_terminal(stage='s4', step='end',
                                             t1=starttime, t2=endtime,
                                             var=self.check_spec_indices)

        self.completed_stages.append('s4')

        # Save the scouse object automatically
        if self.autosave:
            import pickle
            if os.path.exists(self.outputdirectory+self.filename+'/stage_4/s4.scousepy'):
                os.rename(self.outputdirectory+self.filename+'/stage_4/s4.scousepy',self.outputdirectory+self.filename+'/stage_4/s4.scousepy.bk')

            with open(self.outputdirectory+self.filename+'/stage_4/s4.scousepy', 'wb') as fh:
                pickle.dump((self.completed_stages,self.check_spec_indices,self.indiv_dict), fh, protocol=proto)

        return self

    def load_stage_4(self, fn):
        import pickle
        with open(fn, 'rb') as fh:
            self.completed_stages, self.check_spec_indices, self.indiv_dict = pickle.load(fh)

#==============================================================================#
# io
#==============================================================================#

    def run_setup(filename, datadirectory, outputdir=None, description=True,
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

        config_filename='scousepy.config'
        config_filename_coverage='coverage.config'

        scousedir=os.path.join(outputdir, filename)
        configdir=os.path.join(scousedir+'/config_files')
        configpath=os.path.join(scousedir+'/config_files', config_filename)
        configpath_coverage=os.path.join(scousedir+'/config_files', config_filename_coverage)

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
                generate_config_file(filename, datadirectory, outputdir, configdir, config_filename_coverage, description, coverage=True)
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
        # load the cube
        fitsfile = os.path.join(self.datadirectory, self.filename+'.fits')
        self.load_cube(fitsfile=fitsfile)

        return stats(scouseobject=self)
