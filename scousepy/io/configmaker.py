# Licensed under an MIT open source license - see LICENSE

import numpy as np
import os

class ConfigMaker(object):
    """
    Creates a configuration file for scousepy

    Parameters
    ----------


    """
    def __init__(self, config_filename, description=True, configtype='init', dict={}):

        self.config_filename=config_filename
        self.description=description
        self.configtype=configtype
        self.dict=dict

        if self.configtype=='init':

            self.set_defaults_init()
            self.set_defaults_init_desc()

            if os.path.exists(config_filename):
                # initial configuration file for running scousepy
                config_keys=['DEFAULT','stage_1','stage_2','stage_3']
                for config_key in config_keys:
                    self.import_from_config(config_filename,config_key=config_key)

            # if a dictionary is provided set the relevant keywords
            if dict:
                for key, item in dict.items():
                    # first check to see if that attribute exists
                    if hasattr(self, key):
                        if (isinstance(item, str)):
                            item=self.make_string(item)
                        setattr(self, key, item)

        elif self.configtype=='coverage':

            self.set_defaults_cov()
            self.set_defaults_cov_desc()

            if os.path.exists(config_filename):
                config_keys=['DEFAULT']
                self.import_from_config(config_filename)

            # if a dictionary is provided set the relevant keywords
            if dict:
                for key, item in dict.items():
                    # first check to see if that attribute exists
                    if hasattr(self, key):
                        if (isinstance(item, str)):
                            item=self.make_string(item)
                        setattr(self, key, item)

        else:
            pass

        dct_default, dct_stage_1, dct_stage_2, dct_stage_3=self.makedicts()
        self.config_file=self.makeconfig(dct_default, dct_stage_1, dct_stage_2, dct_stage_3)
        with open(os.path.join(self.config_filename), 'w') as file:
            for line in self.config_file:
                file.write(line)

    def set_defaults_init(self):
        self.config_file=str('# ScousePy config file\n\n')
        self.datadirectory=None
        self.filename=None
        self.outputdirectory=None
        self.fittype=self.make_string('Gaussian')
        self.fittype_description="decomposition model (default=Gaussian)"
        self.fittype_simple=True
        self.njobs=3
        self.verbose=True
        self.autosave=True
        self.write_moments=True
        self.save_fig=True
        self.write_ascii=True
        self.tol=[2.0,3.0,1.0,2.5,2.5,0.5]

    def set_defaults_init_desc(self):

        self.datadirectory_description="location of the FITS data cube you would like to decompose"
        self.datadirectory_simple=False
        self.filename_description="name of the FITS data cube (without extension)"
        self.filename_simple=False
        self.outputdirectory_description="output directory for data products"
        self.outputdirectory_simple=False
        self.njobs_description="Number of CPUs used for parallel processing"
        self.njobs_simple=True
        self.verbose_description="print messages to the terminal [True/False]"
        self.verbose_simple=True
        self.autosave_description="autosave output from individual steps [True/False]"
        self.autosave_simple=True
        self.write_moments_description="save moment maps as FITS files [True/False]"
        self.write_moments_simple=False
        self.save_fig_description="generate a figure of the coverage map [True/False]"
        self.save_fig_simple=False
        self.write_ascii_description="outputs an ascii table of the fits [True/False]"
        self.write_ascii_simple=False
        self.tol_description="Tolerance values for the fitting. See Henshaw et al. 2016a"
        self.tol_simple=True

    def set_defaults_cov(self):
        self.config_file=str('# ScousePy config file\n\n')
        self.nrefine=1
        self.mask_below=0.0
        self.mask_coverage=None
        self.x_range=[None, None]
        self.y_range=[None, None]
        self.vel_range=[None, None]
        self.wsaa=[3]
        self.fillfactor=[0.5]
        self.samplesize=0
        self.covmethod=self.make_string('regular')
        self.spacing=self.make_string('nyquist')
        self.speccomplexity=self.make_string('momdiff')
        self.totalsaas=None
        self.totalspec=None


    def set_defaults_cov_desc(self):
        self.nrefine_description="number of refinement steps"
        self.nrefine_simple=True
        self.mask_below_description="mask data below this value"
        self.mask_below_simple=True
        self.mask_coverage_description="optional input filepath to a fits file containing a mask used to define the coverage"
        self.mask_coverage_simple=False
        self.x_range_description="data x range in pixels"
        self.x_range_simple=False
        self.y_range_description="data y range in pixels"
        self.y_range_simple=False
        self.vel_range_description="data velocity range in cube units"
        self.vel_range_simple=False
        self.wsaa_description="width of the spectral averaging areas"
        self.wsaa_simple=True
        self.fillfactor_description="fractional limit below which SAAs are rejected"
        self.fillfactor_simple=True
        self.samplesize_description="sample size for randomly selecting SAAs"
        self.samplesize_simple=True
        self.covmethod_description="method used to define the coverage [regular/random]"
        self.covmethod_simple=True
        self.spacing_description="method setting spacing of SAAs [nyquist/regular]"
        self.spacing_simple=True
        self.speccomplexity_description="method defining spectral complexity"
        self.speccomplexity_simple=True
        self.totalsaas_description="total number of SAAs"
        self.totalsaas_simple=False
        self.totalspec_description="total number of spectra within the coverage"
        self.totalspec_simple=False

    def make_string(self, st):
        newstring="\'" + str(st) + "\'"
        return newstring

    def append_keywords(self, config_file, dct, all_keywords=False, description=True):
        for key in dct.keys():
            if all_keywords:
                if description:
                    config_file.append(
                        '\n\n# {}'.format(dct[key]['description']))
                config_file.append('\n{} = {}'.format(key, dct[key]['default']))
            else:
                if dct[key]['simple']:
                    if description:
                        config_file.append(
                            '\n\n# {}'.format(dct[key]['description']))
                    config_file.append('\n{} = {}'.format(key, dct[key]['default']))
        return config_file

    def makedicts(self):

        if self.configtype=='init':
            default = [
                ('datadirectory', {
                    'default': self.datadirectory,
                    'description': self.datadirectory_description,
                    'simple': self.datadirectory_simple}),
                ('filename', {
                    'default': self.filename,
                    'description': self.filename_description,
                    'simple': self.filename_simple}),
                ('outputdirectory', {
                    'default': self.outputdirectory,
                    'description': self.outputdirectory_description,
                    'simple': self.outputdirectory_simple}),
                ('fittype', {
                    'default': self.fittype,
                    'description': self.fittype_description,
                    'simple': self.fittype_simple}),
                ('njobs', {
                    'default': self.njobs,
                    'description': self.njobs_description,
                    'simple': self.njobs_simple}),
                ('verbose', {
                    'default': self.verbose,
                    'description': self.verbose_description,
                    'simple': self.verbose_simple}),
                ('autosave', {
                    'default': self.autosave,
                    'description': self.autosave_description,
                    'simple': self.autosave_simple}),
                ]

            stage_1 = [
                ('write_moments', {
                    'default': self.write_moments,
                    'description': self.write_moments_description,
                    'simple': self.write_moments_simple}),
                ('save_fig', {
                    'default': self.save_fig,
                    'description': self.save_fig_description,
                    'simple': self.save_fig_simple}),
                ]

            stage_2 = [
                ('write_ascii', {
                    'default': self.write_ascii,
                    'description': self.write_ascii_description,
                    'simple': self.write_ascii_simple}),
                ]

            stage_3 = [
                ('tol', {
                    'default': self.tol,
                    'description': self.tol_description,
                    'simple': self.tol_simple}),
                ]

        elif self.configtype=='coverage':
            default = [
                ('nrefine', {
                    'default': self.nrefine,
                    'description': self.nrefine_description,
                    'simple': self.nrefine_simple}),
                ('mask_below', {
                    'default': self.mask_below,
                    'description': self.mask_below_description,
                    'simple': self.mask_below_simple}),
                ('mask_coverage', {
                    'default': self.mask_coverage,
                    'description': self.mask_coverage_description,
                    'simple': self.mask_coverage_simple}),
                ('x_range', {
                    'default': self.x_range,
                    'description': self.x_range_description,
                    'simple': self.x_range_simple}),
                ('y_range', {
                    'default': self.y_range,
                    'description': self.y_range_description,
                    'simple': self.y_range_simple}),
                ('vel_range', {
                    'default': self.vel_range,
                    'description': self.vel_range_description,
                    'simple': self.vel_range_simple}),
                ('wsaa', {
                    'default': self.wsaa,
                    'description': self.wsaa_description,
                    'simple': self.wsaa_simple}),
                ('fillfactor', {
                    'default': self.fillfactor,
                    'description': self.fillfactor_description,
                    'simple': self.fillfactor_simple}),
                ('samplesize', {
                    'default': self.samplesize,
                    'description': self.samplesize_description,
                    'simple': self.samplesize_simple}),
                ('covmethod', {
                    'default': self.covmethod,
                    'description': self.covmethod_description,
                    'simple':self.covmethod_simple}),
                ('spacing', {
                    'default': self.spacing,
                    'description': self.spacing_description,
                    'simple': self.spacing_simple}),
                ('speccomplexity', {
                    'default': self.speccomplexity,
                    'description': self.speccomplexity_description,
                    'simple': self.speccomplexity_simple}),
                ('totalsaas', {
                    'default': self.totalsaas,
                    'description': self.totalsaas_description,
                    'simple': self.totalsaas_simple}),
                ('totalspec', {
                    'default': self.totalspec,
                    'description': self.totalspec_description,
                    'simple': self.totalspec_simple}),
                ]

            stage_1 = []

            stage_2 = []

            stage_3 = []

        else:
            pass

        from collections import OrderedDict
        dct_default = OrderedDict(default)
        dct_stage_1 = OrderedDict(stage_1)
        dct_stage_2 = OrderedDict(stage_2)
        dct_stage_3 = OrderedDict(stage_3)

        return dct_default, dct_stage_1, dct_stage_2, dct_stage_3

    def makeconfig(self, dct_default, dct_stage_1, dct_stage_2, dct_stage_3):

        config_file = []

        config_file.append('[DEFAULT]')
        config_file = self.append_keywords(config_file, dct_default,
                                            all_keywords=True,
                                            description=self.description)

        if self.configtype=='init':
            config_file.append('\n\n[stage_1]')
            config_file = self.append_keywords(config_file, dct_stage_1,
                                                all_keywords=True,
                                                description=self.description)

            config_file.append('\n\n[stage_2]')
            config_file = self.append_keywords(config_file, dct_stage_2,
                                                all_keywords=True,
                                                description=self.description)

            config_file.append('\n\n[stage_3]')
            config_file = self.append_keywords(config_file, dct_stage_3,
                                                all_keywords=True,
                                                description=self.description)

        return config_file

    def import_from_config(self, config_file, config_key='DEFAULT'):
        """
        Read in values from configuration table.

        Parameters
        ----------
        config_file : str
            Filepath to configuration file
        config_key : str
            Section of configuration file, whose parameters should be read in addition to 'DEFAULT'.

        Notes
        -----
        adapted from Gausspy+ methodology

        """
        import ast
        import configparser

        config = configparser.ConfigParser()
        config.read(config_file)

        for key, value in config[config_key].items():
            try:
                value=ast.literal_eval(value)
                if isinstance(value, str):
                    setattr(self, key, self.make_string(value))
                else:
                    setattr(self, key, value)
            except ValueError:
                raise Exception('Could not parse parameter {} from config file'.format(key))
