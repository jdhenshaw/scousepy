# Licensed under an MIT open source license - see LICENSE

from astropy import units as u
from astropy.io import fits
from spectral_cube import SpectralCube
from astropy import wcs
from astropy.table import Table
from astropy.table import Column
from astropy import log
from astropy.utils.console import ProgressBar
import numpy as np
import os
import sys
import warnings
import pyspeckit
import shutil
import time
import pickle

from .parallel_map import *

if sys.version_info.major >= 3:
    proto=3
else:
    proto=2

def create_directory_structure(scousedir):
    """
    Make the output directory
    """
    from .colors import colors

    if not os.path.exists(scousedir):
        os.makedirs(scousedir)
        mkdirectory(os.path.join(scousedir, 'stage_1'))
        mkdirectory(os.path.join(scousedir, 'stage_2'))
        mkdirectory(os.path.join(scousedir, 'stage_3'))
        mkdirectory(os.path.join(scousedir, 'stage_4'))
        mkdirectory(os.path.join(scousedir, 'config_files'))
    else:
        pass

def mkdirectory(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        pass

def append_keywords(config_file, dct, all_keywords=False, description=True):
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

def make_string(st):
    newstring="\'" + str(st) + "\'"
    return newstring

def generate_config_file(filename, datadirectory, outputdir, configdir, config_filename, description, coverage=False, dict=None, covdict=None):
    """
    Creates the configuration table for scousepy

    Parameters
    ----------
    filename : string
        output filename of the config file
    datadirectory : string
        directory containing the datacube
    outputdir : string
        output directory
    configdir : string
        directory containing the config files
    config_filename : string
        filename of the configuration file
    description : bool
        whether or not to include the description of each parameter

    Notes
    -----
    adapted from Gausspy+ methodology

    """
    from collections import OrderedDict

    config_file = str('# ScousePy config file\n\n')

    if not coverage:
        default = [
            ('datadirectory', {
                'default': make_string(datadirectory),
                'description': "location of the FITS data cube you would like to decompose",
                'simple': True}),
            ('filename', {
                'default': make_string(filename),
                'description': "name of the FITS data cube (without extension)",
                'simple': True}),
            ('outputdirectory', {
                'default': make_string(outputdir),
                'description': "output directory for data products",
                'simple': True}),
            ('fittype', {
                'default': make_string('gaussian'),
                'description': "decomposition model (default=Gaussian)",
                'simple': True}),
            ('njobs', {
                'default': '3',
                'description': "Number of CPUs used for parallel processing",
                'simple': True}),
            ('verbose', {
                'default': 'True',
                'description': "print messages to the terminal [True/False]",
                'simple': True}),
            ('autosave', {
                'default': 'True',
                'description': "autosave output from individual steps [True/False]",
                'simple': True}),
            ]

    else:
        default = [
            ('nrefine', {
                'default': '1',
                'description': "number of refinement steps",
                'simple': True}),
            ('mask_below', {
                'default': '0.0',
                'description': "mask data below this value",
                'simple': True}),
            ('mask_coverage', {
                'default': 'None',
                'description': "optional input filepath to a fits file containing a mask used to define the coverage",
                'simple': False}),
            ('x_range', {
                'default': "[None, None]",
                'description': "data x range in pixels",
                'simple': False}),
            ('y_range', {
                'default': "[None, None]",
                'description': "data y range in pixels",
                'simple': False}),
            ('vel_range', {
                'default': "[None, None]",
                'description': "data velocity range in cube units",
                'simple': False}),
            ('wsaa', {
                'default': '[3]',
                'description': "width of the spectral averaging areas",
                'simple': True}),
            ('fillfactor', {
                'default': "[0.5]",
                'description': "fractional limit below which SAAs are rejected",
                'simple': True}),
            ('samplesize', {
                'default': '0',
                'description': "sample size for randomly selecting SAAs",
                'simple': True}),
            ('covmethod', {
                'default': make_string('regular'),
                'description': "method used to define the coverage [regular/random]",
                'simple': True}),
            ('spacing', {
                'default': make_string('nyquist'),
                'description': "method setting spacing of SAAs [nyquist/regular]",
                'simple': True}),
            ('speccomplexity', {
                'default': make_string('momdiff'),
                'description': "method defining spectral complexity",
                'simple': True}),
            ('totalsaas', {
                'default': 'None',
                'description': "total number of SAAs",
                'simple': False}),
            ('totalspec', {
                'default': 'None',
                'description': "total number of spectra within the coverage",
                'simple': False}),
            ]


    stage_1 = [
        ('write_moments', {
            'default': 'True',
            'description': "save moment maps as FITS files [True/False]",
            'simple': False}),
        ('save_fig', {
            'default': 'True',
            'description': "generate a figure of the coverage map [True/False]",
            'simple': False}),
        ]

    stage_2 = [
        ('write_ascii', {
            'default': 'True',
            'description': "outputs an ascii table of the fits [True/False]",
            'simple': False}),
        ]

    stage_3 = [
        ('tol', {
            'default': "[2.0,3.0,1.0,2.5,2.5,0.5]",
            'description': "Tolerance values for the fitting. See Henshaw et al. 2016a",
            'simple': True}),
        ]

    dct_default = OrderedDict(default)
    dct_stage_1 = OrderedDict(stage_1)
    dct_stage_2 = OrderedDict(stage_2)
    dct_stage_3 = OrderedDict(stage_3)

    config_file = []

    config_file.append('[DEFAULT]')
    config_file = append_keywords(config_file, dct_default,
                                  all_keywords=True,
                                  description=description)

    if not coverage:
        config_file.append('\n\n[stage_1]')
        config_file = append_keywords(config_file, dct_stage_1,
                                      all_keywords=True,
                                      description=description)

        config_file.append('\n\n[stage_2]')
        config_file = append_keywords(config_file, dct_stage_2,
                                      all_keywords=True,
                                      description=description)

        config_file.append('\n\n[stage_3]')
        config_file = append_keywords(config_file, dct_stage_3,
                                      all_keywords=True,
                                      description=description)

    with open(os.path.join(configdir, config_filename), 'w') as file:
        for line in config_file:
            file.write(line)

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
            setattr(self, key, value)
        except ValueError:
            raise Exception('Could not parse parameter {} from config file'.format(key))

def write_averaged_spectra(cube_header, saa_spectra, r, dir,
                           fits_fmatter='saa_cube_r{}.fits'):
    """
    Writes spectra averaged on multiple scales into fits files.

    Parameters
    ----------
    cube_header : FITS header of the original spectral cube

    saa_spectra : len(N) list
                  Contains spectra averaged over N scales

    wsaa : len(N) list
           List of averaging radii

    fits_fmatter : a string formatter for output files to be written to
    """

    #for aver_cube in :
    hdu = fits.PrimaryHDU(data=saa_spectra, header=cube_header)
    hdu.header['wsaa'] = r
    hdu.writeto(dir+'/saa_cube_r_{}.fits'.format(r), overwrite=True)

def output_moments(cube_header, moments, dir, filename):
    """
    Write the moment maps out as fits files
    """
    try:
        unit=cube_header['BUNIT']
    except KeyError:
        cube_header['BUNIT']='I'
        unit=cube_header['BUNIT']

    header=moments[0].header

    for i in range(len(moments)-1):

        if i == 0:
            myunit=unit+str(' km s-1')
        elif (i==1) or (i==2) or (i==5):
            myunit=str('km s-1')
        else:
            myunit=''

        header['BUNIT']=myunit
        if (i==0) or (i==1) or (i==2):
            name='_mom'+str(i)
            fits.writeto(dir+filename+name+'.fits', moments[i].value, header, overwrite=True)
        elif (i==5):
            name='_velatpeak'
            fits.writeto(dir+filename+name+'.fits', moments[i].value, header, overwrite=True)
        else:
            pass
            # FOR NOW MOM 3 and 4 ARE REMOVED

            #name='_mom'+str(i)
            #fits.writeto(dir+filename+name+'.fits', moments[i], header, overwrite=True)

def output_ascii_saa(self, path):
    """
    Outputs an ascii table containing the information for each fit.
    """

    for i in range(len(self.wsaa)):
        saa_dict = self.saa_dict[i]
        if any(SAA.to_be_fit for SAA in saa_dict.values()):
            table = make_table(self, saa_dict, saa=True)
            table.write(path+'_model_solutions_'+str(self.wsaa[i])+'.dat', format='ascii', \
                        overwrite=True, delimiter='\t')
        else:
            # I don't know how we got to this case, but I encountered it at least once.
            print("No fits were found for the {0}'th spectral averaging area.".format(i))

def output_ascii_indiv(self, outputdir):
    """
    Outputs an ascii table containing the information for each fit.
    """

    saa_dict = self.saa_dict[0]
    table = make_table(self, saa_dict, indiv=True)
    table.write(outputdir+'/best_fit_solutions.dat', format='ascii', \
                overwrite=True, delimiter='\t')
    return

def make_table(self, saa_dict, saa=False, indiv=False):
    """
    Generates an astropy table to hold the information
    """

    table = Table(meta={'name': 'best-fitting model solutions'})
    headings = get_headings(self, saa_dict)

    if saa:
        solnlist = get_solnlist_saa(self, saa_dict)
    elif indiv:
        solnlist = get_solnlist_indiv(self)

    solnarr = np.asarray(solnlist).T

    for j in range(len(solnarr[:,0])):
        table[headings[j]] = Column(solnarr[j,:])

    return table

def get_headings(self, saa_dict):
    """
    Table headings for output

    Notes:

    This is awful but it works.
    """

    cont = True
    keys = list(saa_dict.keys())
    count = 0

    headings = []
    # find the first spectral averaging area where there is a fit
    # and get the parameter names
    while cont:
        SAA = saa_dict[keys[count]]
        if SAA.to_be_fit:
            soln = SAA.model
            # These headings never change
            headings_non_specific = ['ncomps', 'x', 'y', 'rms', 'residual', \
                                      'chisq', 'dof', 'redchisq', 'aic' ]
            #These ones depend on the model used by pyspeckit
            headings_params = soln.parnames
            headings_errs = [str('err {0}').format(soln.parnames[k]) for k in range(len(soln.parnames))]
            # This is messy
            headings_pars = [[headings_params[k], headings_errs[k]] for k in range(len(soln.parnames))]
            headings_pars = [par for pars in headings_pars for par in pars]
            headings = headings_non_specific[0:3]+headings_pars+headings_non_specific[3::]

            cont=False
            count+=1
        else:
            count+=1

    return headings

def get_solnlist_saa(self, saa_dict):
    """
    Returns list of SAA solutions
    """
    solnlist = []
    #for j in range(len(saa_dict.keys())):
    for key in saa_dict.keys():
        # get the relavent SAA
        #SAA = saa_dict[j]
        SAA = saa_dict[key]

        if SAA.to_be_fit:
            soln = SAA.model
            for k in range(int(soln.ncomps)):
                solution_desc = get_soln_desc(k, soln, \
                                              SAA.coordinates[0], \
                                              SAA.coordinates[1])
                solnlist.append(solution_desc)

    return solnlist

def get_solnlist_indiv(self):
    """
    Returns list of solutions to individual pixels
    """
    solnlist = []
    for key in self.indiv_dict.keys():
        spectrum = self.indiv_dict[key]
        soln = spectrum.model
        if soln.ncomps==0.0:
            solution_desc = get_soln_desc(0, soln, \
                                          spectrum.coordinates[0], \
                                          spectrum.coordinates[1])
            solnlist.append(solution_desc)
        else:
            for k in range(int(soln.ncomps)):
                solution_desc = get_soln_desc(k, soln, \
                                              spectrum.coordinates[0], \
                                              spectrum.coordinates[1])
                solnlist.append(solution_desc)

    return solnlist

def get_soln_desc(idx, soln, x, y):
    """
    Returns the solution in the format:

    ncomps, x, y, param1, err1, .... paramn, errn, rms, residstd, chi2, dof,
    chi2red, aic
    """
    params_non_specific = [soln.ncomps, \
                           x, y,\
                           soln.rms, soln.residstd, soln.chisq, \
                           soln.dof, soln.redchisq, soln.AIC]

    parlow = int((idx*len(soln.parnames)))
    parhigh = int((idx*len(soln.parnames))+len(soln.parnames))
    parrange = np.arange(parlow,parhigh)

    paramarr = np.array(soln.params)
    errarr = np.array(soln.errors)
    params = paramarr[parrange]
    errors = errarr[parrange]
    # This is messy
    parameters = [[params[j], errors[j]] for j in range(len(soln.parnames))]
    parameters = [par for pars in parameters for par in pars]

    solution_desc = params_non_specific[0:3]+parameters+params_non_specific[3::]

    return solution_desc

def create_modelcube(self, njobs=1, verbose=True):
    """
    Generates a "clean" datacube from the scousepy decomposition. Returns a
    clean cube

    Parameters
    ----------
    self : instance of the scousepy class
    njobs : Number
        number of cpus
    verbose: bool
        verbose output

    """

    # Time it
    starttime = time.time()

    cube = self.cube
    x = np.array(cube.world[:,0,0][0])
    if (self.ppv_vol[0] is not None) & (self.ppv_vol[1] is not None):
        trimids = np.where((x>self.ppv_vol[0])&(x<self.ppv_vol[1]))[0]

    _cube = cube[min(trimids):max(trimids)+1, :, :]
    _modelcube = np.full_like(_cube, np.nan)

    if verbose:
        print("")
        print("Generating models:")
        print("")

    args = [self]
    inputs = [[key] + args for key in self.indiv_dict.keys()]
    if njobs==1:
        mods = ProgressBar.map(genmodel, inputs)
    else:
        mods = parallel_map(genmodel, inputs, numcores=njobs)
    mergedmods = [mod for mod in mods]
    mergedmods = np.asarray(mergedmods)

    if verbose:
        print("")
        print("Creating model cube:")
        print("")
        progress_bar = ProgressBar(self.indiv_dict.keys())

    for i, key in enumerate(self.indiv_dict.keys()):
        _modelcube[:, self.indiv_dict[key].coordinates[0],
                      self.indiv_dict[key].coordinates[1]] = mergedmods[i]
        if verbose:
            progress_bar.update()

    endtime = time.time()
    if verbose:
        print("")
        print('Process completed in: {0} minutes'.format((endtime-starttime)/60.))
        print("")

    return SpectralCube(data=_modelcube, wcs=_cube.wcs)

def genmodel(inputs):
    """
    generates the model for the creation of the model cube
    """
    key, self = inputs
    spectrum = self.indiv_dict[key]
    bfmodel = spectrum.model
    if bfmodel.ncomps>0:
        from .stage_5 import recreate_model
        mod,res = recreate_model(self, spectrum, bfmodel)
        totmod = np.nansum(mod, axis=1)
    else:
        totmod = np.full_like(self.xtrim, np.nan)

    return totmod

def save(self, filename):
    """
    Saves the output file - requires pickle.
    """
    pickle.dump( self, open( filename, "wb" ), protocol=proto )

def load(filename):
    """
    Loads a previously computed file - requires pickle.
    """
    loadedfile = pickle.load( open(filename, "rb"))
    return loadedfile
