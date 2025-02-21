import os
import time

from astropy import log

from .utils.logger import catch_warnings, warning_to_log


class scouse(object):
    """the scouse class

    Attributes
    ----------
    config : str, optional
        Path to the configuration file. This must be provided.
    datadirectory : str
        Directory containing the datacube
    filename : str
        Name of the file to be loaded
    outputdirectory : str
        Output directory for data products
    fittype : str
        Decomposition model (default=Gaussian)
    verbose : bool
        Print messages to the terminal
    autosave : bool
        Autosave output from individual steps
    cube : astropy.io.fits.HDUList
        The datacube
    completed_stages : list
        List of completed stages
    write_moments : bool
        Write moments to a file
    save_fig : bool
        Save figures to a file
    coverage_config_file_path : str
        Path to the coverage configuration file
    nrefine : int
        Number of refinement steps
    mask_below : float
        Mask below this value
    x_range : tuple
        Spatial range to fit
    y_range : tuple
        Spatial range to fit
    vel_range : tuple
        Velocity range to fit
    wsaa : float
        Width of the spatial average
    fillfactor : float
        Fill factor for the spatial average
    samplesize : int
        Number of spectra to sample
    covmethod : str
        Method for calculating the coverage
    spacing : float
        Spacing of the spatial average
    speccomplexity : float
        Complexity of the spectra
    totalsaas : int
        Total number of SAAs
    totalspec : int
        Total number of spectra
    lenspec : int
        Number of spectra in the datacube
    saa_dict : dict
        Dictionary of SAAs
    rms_approx : float
        RMS approximation
    x : array
        Spatial coordinates
    xtrim : array
        Trimmed spatial coordinates
    trimids : array
        Indices of the trimmed spectra
    chunk : bool
        Whether to chunk the datacube
    write_ascii : bool
        Write ASCII output
    fitcount : int
        Number of fits
    modelstore : dict
        Store of models
    tol : float
        Tolerance for the fit
    njobs : int
        Number of jobs to run in parallel
    check_spec_indices : list
        Indices of the spectra to check

    Methods
    -------
    run_setup : str
        Generates a scousepy configuration file
    stage_1 : None
        Identify the spatial area over which the fitting will be implemented.
    stage_2 : None
        Fit the spectra in the selected area.
    stage_3 : None
        Refine the fit to the spectra.
    stage_4 : None
        Save the results.

    """

    def __init__(self, config: str | None = None):
        """
        Initialise the scouse class

        Parameters
        ----------
        config : str, optional
            Path to the configuration file. This must be provided.
        """

        self.config = config
        self.config_coverage = ""
        self.logfile_path = ""
        self.datadirectory = ""
        self.filename = ""
        self.outputdirectory = ""
        self.fittype = ""
        self.verbose = None
        self.autosave = None
        self.cube = None
        self.completed_stages = []

    @catch_warnings
    def run_setup(
        filename: str,
        datadirectory: str,
        outputdir: str | None = None,
        description: bool = True,
        verbose: bool = True,
    ):
        """
        Generates a scousepy configuration file

        Parameters
        ----------
        filename : str
            Name of the file to be loaded
        datadirectory : str
            Directory containing the datacube
        outputdir : str, optional
            Alternate output directory. Deflault is datadirectory
        description : bool, optional
            whether or not a description of each parameter is included in the
            configuration file
        verbose : bool, optional
            verbose output to terminal

        Returns
        -------
        configpath : str
            Path to the configuration file

        """
        from .io.config import Config
        from .utils.logger import Logger

        if outputdir is None:
            outputdir = datadirectory

        config_filename = "scousepy.config"
        config_filename_coverage = "coverage.config"
        logfile = os.path.join(outputdir, filename, "scousepy.log")
        configdir = os.path.join(outputdir, filename, "config_files")
        configpath = os.path.join(outputdir, filename, "config_files", config_filename)

        # Set up logging
        logger = Logger(logfile=logfile)
        logger.printlog(
            message="initializing scousepy", status="staging", print_to_terminal=verbose
        )

        if os.path.exists(configpath):
            logger.printlog(
                message="scousepy configuration file already exists. ",
                status="severe",
            )
            return configdir
        else:
            if not os.path.exists(os.path.join(outputdir, filename, "config_files")):
                Config.create_directory_structure(os.path.join(outputdir, filename))
                Config.generate_config_file(
                    filename,
                    datadirectory,
                    outputdir,
                    configdir,
                    config_filename,
                    description,
                )
                Config.generate_config_file(
                    filename,
                    datadirectory,
                    outputdir,
                    configdir,
                    config_filename_coverage,
                    description,
                    coverage=True,
                )
                logger.printlog(
                    message="scousepy configuration files created. ",
                    status="pass",
                    print_to_terminal=verbose,
                )
            else:
                configpath = None
                logger.printlog(
                    message="output directory exists but does not contain a config file. ",
                    status="severe",
                )

        return configdir

    @staticmethod
    @catch_warnings
    def stage_1(
        config: str = "",
        interactive: bool = True,
        verbose: bool = True,
        nchunks: int | None = None,
        s1file: str | None = None,
        max_samples: int = 500,
        fail_limit: int = 50,
    ):
        """
        Identify the spatial area over which the fitting will be implemented.

        Parameters
        ----------
        config : str, optional
            Path to the configuration file. This must be provided.
        interactive : bool, optional
            Default is to run coverage with interactive GUI, but this can be
            bypassed in favour of using the config file
        verbose : bool, optional
            Default is to print to the terminal, but this can be bypassed
        nchunks : int, optional
            Number of chunks to split the cube into
        s1file : str, optional
            Path to the file containing the spatial coverage
        max_samples : int, optional
            The maximum number of samples to use to estimate the noise.
        fail_limit : int, optional
            The maximum number of times the noise estimation can fail before
            the function returns the median of the MAD of the spectrum.

        """
        from .io.config import Config
        from .io.cubehandler import CubeHandler
        from .io.stageio import LoadScouseObject, SaveScouseObject
        from .processing.SAAs import SAAs
        from .utils.logger import Logger

        self = scouse(config=config)
        Config.load_config_files(self, config, ["stage_1"])

        logfile = os.path.join(Config.get_scousedir(self.config), "scousepy.log")
        logger = Logger(logfile=logfile)

        self.verbose = True if verbose else False

        # check if stage 1 has already been run
        if s1file is not None:
            s1path = self.outputdirectory + self.filename + "/stage_1/" + s1file
        else:
            s1path = self.outputdirectory + self.filename + "/stage_1/s1.scousepy"

        # check if the stage 1 file exists. If so, load it.
        if os.path.exists(s1path):
            logger.printlog(
                message="loading s1.scousepy",
                status="staging",
                print_to_terminal=self.verbose,
            )
            self = LoadScouseObject.load_stage_1(self, s1path)
            if "s1" in self.completed_stages:
                logger.printlog(
                    message="coverage complete and SAAs initialised",
                    status="pass",
                    print_to_terminal=self.verbose,
                )

            return self

        sys.exit()

        # load the cube
        fitsfile = os.path.join(self.datadirectory, self.filename)
        self.cube = CubeHandler.load_cube(fitsfile)

        logger.printlog(
            message="beginning stage 1 analysis",
            status="staging",
            print_to_terminal=self.verbose,
        )

        with warning_to_log():
            old_log = log.level
            log.setLevel("ERROR")

            # Interactive coverage generator
            if interactive:
                # if interactive is True, then we are going to use the SAAsGUI to generate the coverage SAAs
                from .interative.SAAsGUI import SAAsGUI

                gui = SAAsGUI(scouseobject=self, verbose=self.verbose)
                gui.show()
                self = gui.scouseobject
                coverageobject = SAAs(scouseobject=self)
                coverageobject.establishSAAs()
            else:
                # if interactive is False, then we are going to assume that the coverage config file has been modified and we are going to use the values from the coverage config file and use these to define the coverage SAAs
                coverageobject = SAAs(scouseobject=self)
                coverageobject.establishSAAs()

        # start the time once the coverage has been generated
        starttime = time.time()

        # Create a dictionary to store the SAAs
        self.saa_dict = {}
        # Compute typical noise within the spectra
        self.rms_approx = CubeHandler.estimate_noise(
            self.cube, max_samples=max_samples, fail_limit=fail_limit
        )
        # Generate the x axis common to the fitting process
        self.x, self.xtrim, self.trimids = CubeHandler.get_x_axis(self)

        # SAAs
        with warning_to_log():
            old_log = log.level
            log.setLevel("ERROR")
            pass
            log.setLevel(old_log)

        # wrapping up
        endtime = time.time()
        logger.printlog(
            message="stage 1 complete",
            status="pass",
            timetaken=endtime - starttime,
            print_to_terminal=bool(self.verbose),
        )
        self.completed_stages.append("s1")

        if s1file is not None:
            s1path = self.outputdirectory + self.filename + "/stage_1/" + s1file
            SaveScouseObject.save_stage_1(self, s1path)
        else:
            s1path = self.outputdirectory + self.filename + "/stage_1/s1.scousepy"
            SaveScouseObject.save_stage_1(self, s1path)

        return self
