from os import makedirs, path


class Config:
    """
    Initialize the scousepy directory structure and configuration file
    """

    @staticmethod
    def create_directory_structure(scousedir: str) -> None:
        """Create the directory structure for scousepy

        Parameters
        ----------
        scousedir : str
            The directory to create the structure in

        """

        if not path.exists(path.join(scousedir, "config_files")):
            makedirs(path.join(scousedir, "config_files"))
            makedirs(path.join(scousedir, "stage_1"))
            makedirs(path.join(scousedir, "stage_2"))
            makedirs(path.join(scousedir, "stage_3"))
            makedirs(path.join(scousedir, "stage_4"))
        else:
            pass

    @staticmethod
    def _make_string(st):
        """Make a string a string

        Parameters
        ----------
        st : str
            The string to make
        """
        newstring = "'" + str(st) + "'"
        return newstring

    @staticmethod
    def _append_keywords(
        config_file: list,
        dct: dict,
        all_keywords: bool = True,
        description: bool = False,
    ):
        """Append keywords to the config file

        Parameters
        ----------
        config_file : list
            The config file to append the keywords to
        dct : dict
            The dictionary of keywords to append
        all_keywords : bool, optional
            Whether to append all keywords (default is True)
        description : bool, optional
            Whether to include a description of each parameter (default is False)

        """
        for key in dct.keys():
            if all_keywords:
                if description:
                    config_file.append("\n\n# {}".format(dct[key]["description"]))
                config_file.append("\n{} = {}".format(key, dct[key]["default"]))
            else:
                if dct[key]["simple"]:
                    if description:
                        config_file.append("\n\n# {}".format(dct[key]["description"]))
                    config_file.append("\n{} = {}".format(key, dct[key]["default"]))
        return config_file

    @staticmethod
    def generate_config_file(
        filename: str,
        datadirectory: str,
        outputdir: str,
        configdir: str,
        config_filename: str,
        description: bool,
        coverage: bool = False,
        dict: dict | None = None,
        covdict: dict | None = None,
    ):
        """Generate a scousepy configuration file

        Parameters
        ----------
        filename : str
            The name of the file to create
        datadirectory : str
            The directory containing the datacube
        outputdir : str
            The directory to output the files to (default is datadirectory)
        configdir : str
            The directory to output the config files to
        config_filename : str
            The name of the config file to create
        description : bool
            Whether to include a description of each parameter
        coverage : bool, optional
            Whether to generate a coverage file
        dict : dict, optional
            A dictionary of parameters for the config file
        covdict : dict, optional
            A dictionary of parameters for the coverage file

        Notes
        -----
        adapted from Gausspy+ (Riener et al. 2019) methodology

        """

        from collections import OrderedDict

        config_file = str("# ScousePy config file\n\n")

        if not coverage:
            default = [
                (
                    "datadirectory",
                    {
                        "default": Config._make_string(datadirectory),
                        "description": "location of the FITS data cube you would like to decompose",
                        "simple": True,
                    },
                ),
                (
                    "filename",
                    {
                        "default": Config._make_string(filename),
                        "description": "name of the FITS data cube (without extension)",
                        "simple": True,
                    },
                ),
                (
                    "outputdirectory",
                    {
                        "default": Config._make_string(outputdir),
                        "description": "output directory for data products",
                        "simple": True,
                    },
                ),
                (
                    "fittype",
                    {
                        "default": Config._make_string("gaussian"),
                        "description": "decomposition model (default=Gaussian)",
                        "simple": True,
                    },
                ),
                (
                    "njobs",
                    {
                        "default": "3",
                        "description": "Number of CPUs used for parallel processing",
                        "simple": True,
                    },
                ),
                (
                    "verbose",
                    {
                        "default": "True",
                        "description": "print messages to the terminal [True/False]",
                        "simple": True,
                    },
                ),
                (
                    "autosave",
                    {
                        "default": "True",
                        "description": "autosave output from individual steps [True/False]",
                        "simple": True,
                    },
                ),
            ]
        else:
            default = [
                (
                    "nrefine",
                    {
                        "default": "1",
                        "description": "number of refinement steps",
                        "simple": True,
                    },
                ),
                (
                    "mask_below",
                    {
                        "default": "0.0",
                        "description": "mask data below this value",
                        "simple": True,
                    },
                ),
                (
                    "mask_coverage",
                    {
                        "default": "None",
                        "description": "optional input filepath to a fits file containing a mask used to define the coverage",
                        "simple": False,
                    },
                ),
                (
                    "x_range",
                    {
                        "default": "[None, None]",
                        "description": "data x range in pixels",
                        "simple": False,
                    },
                ),
                (
                    "y_range",
                    {
                        "default": "[None, None]",
                        "description": "data y range in pixels",
                        "simple": False,
                    },
                ),
                (
                    "vel_range",
                    {
                        "default": "[None, None]",
                        "description": "data velocity range in cube units",
                        "simple": False,
                    },
                ),
                (
                    "wsaa",
                    {
                        "default": "[3]",
                        "description": "width of the spectral averaging areas",
                        "simple": True,
                    },
                ),
                (
                    "fillfactor",
                    {
                        "default": "[0.5]",
                        "description": "fractional limit below which SAAs are rejected",
                        "simple": True,
                    },
                ),
                (
                    "samplesize",
                    {
                        "default": "0",
                        "description": "sample size for randomly selecting SAAs",
                        "simple": True,
                    },
                ),
                (
                    "covmethod",
                    {
                        "default": Config._make_string("regular"),
                        "description": "method used to define the coverage [regular/random]",
                        "simple": True,
                    },
                ),
                (
                    "spacing",
                    {
                        "default": Config._make_string("nyquist"),
                        "description": "method setting spacing of SAAs [nyquist/regular]",
                        "simple": True,
                    },
                ),
                (
                    "speccomplexity",
                    {
                        "default": Config._make_string("momdiff"),
                        "description": "method defining spectral complexity",
                        "simple": True,
                    },
                ),
                (
                    "totalsaas",
                    {
                        "default": "None",
                        "description": "total number of SAAs",
                        "simple": False,
                    },
                ),
                (
                    "totalspec",
                    {
                        "default": "None",
                        "description": "total number of spectra within the coverage",
                        "simple": False,
                    },
                ),
            ]

        stage_1 = [
            (
                "write_moments",
                {
                    "default": "True",
                    "description": "save moment maps as FITS files [True/False]",
                    "simple": False,
                },
            ),
            (
                "save_fig",
                {
                    "default": "True",
                    "description": "generate a figure of the coverage map [True/False]",
                    "simple": False,
                },
            ),
        ]

        stage_2 = [
            (
                "write_ascii",
                {
                    "default": "True",
                    "description": "outputs an ascii table of the fits [True/False]",
                    "simple": False,
                },
            )
        ]

        stage_3 = [
            (
                "tol",
                {
                    "default": "[2.0,3.0,1.0,2.5,2.5,0.5]",
                    "description": "Tolerance values for the fitting. See Henshaw et al. 2016a",
                    "simple": True,
                },
            )
        ]
        dct_default = OrderedDict(default)
        dct_stage_1 = OrderedDict(stage_1)
        dct_stage_2 = OrderedDict(stage_2)
        dct_stage_3 = OrderedDict(stage_3)

        config_file = []

        config_file.append("[DEFAULT]")
        config_file = Config._append_keywords(
            config_file, dct_default, all_keywords=True, description=description
        )

        if not coverage:
            config_file.append("\n\n[stage_1]")
            config_file = Config._append_keywords(
                config_file, dct_stage_1, all_keywords=True, description=description
            )

            config_file.append("\n\n[stage_2]")
            config_file = Config._append_keywords(
                config_file, dct_stage_2, all_keywords=True, description=description
            )

            config_file.append("\n\n[stage_3]")
            config_file = Config._append_keywords(
                config_file, dct_stage_3, all_keywords=True, description=description
            )

        Config.write_config_file(config_file, configdir, config_filename)

    @staticmethod
    def write_config_file(config_file: list, configdir: str, config_filename: str):
        """Write the configuration file to the config directory

        Parameters
        ----------
        config_file : list
            The configuration file to write
        configdir : str
            The directory to write the configuration file to
        config_filename : str
            The name of the configuration file to write
        """
        with open(path.join(configdir, config_filename), "w") as file:
            for line in config_file:
                file.write(line)

    @staticmethod
    def load_config_files(scouseobject, configdir: str, stages: list):
        """Load the configuration files into the scouse object

        Parameters
        ----------
        scouseobject : scouse
            The scouse object to load the configuration files into
        configdir : str
            The directory to load the configuration files from
        stages : list
            The stages to load the configuration files for

        """
        from ..utils.logger import Logger

        scouseobject.config = path.join(configdir, "scousepy.config")
        scouseobject.config_coverage = path.join(configdir, "coverage.config")

        if path.exists(scouseobject.config):
            for stage in stages:
                Config.import_from_config(
                    scouseobject, scouseobject.config, config_key=stage
                )

            logger = Logger(
                logfile=path.join(
                    scouseobject.outputdirectory, scouseobject.filename, "scousepy.log"
                )
            )
            logger.printlog(
                message="configuration files loaded",
                status="pass",
                print_to_terminal=False,
            )
            if path.exists(scouseobject.config_coverage):
                Config.import_from_config(scouseobject, scouseobject.config_coverage)
                logger.printlog(
                    message="coverage config file loaded",
                    status="pass",
                    print_to_terminal=False,
                )
        else:
            logger = Logger()
            logger.printlog(
                message="no scousepy config file found",
                status="severe",
            )
            return

        return scouseobject

    @staticmethod
    def import_from_config(scouseobject, config_file: str, config_key: str = "DEFAULT"):
        """Import the configuration file into the ScouseObject

        Parameters
        ----------
        scouseobject : scouse
            The scouse object to import the configuration file into
        config_file : str
            The configuration file to import
        config_key : str, optional
            The key to import from the configuration file
        """

        import configparser
        from ast import literal_eval

        from ..utils.logger import Logger

        logger = Logger(
            logfile=path.join(Config.get_scousedir(scouseobject.config), "scousepy.log")
        )

        config = configparser.ConfigParser()
        config.read(config_file)

        for key, value in config[config_key].items():
            try:
                value = literal_eval(value)
                setattr(scouseobject, key, value)
            except ValueError:
                logger.printlog(
                    message="Could not parse parameter {} from config file".format(key),
                    status="severe",
                )
                raise Exception()

        return scouseobject

    @staticmethod
    def get_scousedir(config_file: str) -> str:
        """Get the scousedir from the configuration file

        Parameters
        ----------
        config_file : str
            The configuration file to get the scousedir from
        """
        import configparser

        config = configparser.ConfigParser()
        config.read(config_file)

        # Get the outputdirectory directly from the config
        try:
            return config["DEFAULT"]["outputdirectory"].strip("'\"")
        except KeyError:
            try:
                return config["DEFAULT"]["datadirectory"].strip("'\"")
            except KeyError:
                raise KeyError(
                    "Neither 'outputdirectory' nor 'datadirectory' found in config file"
                )
