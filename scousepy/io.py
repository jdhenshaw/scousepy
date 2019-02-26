# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

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

def mkdir_s1(outputdir, s1dir):
    """
    Make the output directory for stage 1
    """
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    if not os.path.exists(s1dir):
        os.mkdir(s1dir)

def mkdir_s2(outputdir, s2dir):
    """
    Make the output directory for stage 2
    """
    if not os.path.exists(s2dir):
        os.makedirs(s2dir)
    else:
        # TODO: error handling
        pass

def mkdir_s3(outputdir, s3dir):
    """
    Make the output directory for stage 3
    """
    if not os.path.exists(s3dir):
        os.makedirs(s3dir)
    else:
        # TODO: error handling
        pass

def mkdir_s4(outputdir, s4dir):
    """
    Make the output directory for stage 4
    """
    if not os.path.exists(s4dir):
        os.makedirs(s4dir)
    else:
        # TODO: error handling
        pass

def mkdir_s5(outputdir, s5dir):
    """
    Make the output directory for stage 5
    """
    if not os.path.exists(s5dir):
        os.makedirs(s5dir)
    else:
        # TODO: error handling
        pass

def mkdir_s6(outputdir, s6dir):
    """
    Make the output directory for stage 6
    """
    if not os.path.exists(s6dir):
        os.makedirs(s6dir)
    else:
        # TODO: error handling
        pass

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

def output_moments(momzero, momone, momtwo, momnine, dir, filename):
    """
    Write the moment maps to file
    """
    momzero.write(dir+'/'+filename+'_momzero.fits', format='fits', overwrite=True)
    momone.write(dir+'/'+filename+'_momone.fits', format='fits', overwrite=True)
    momtwo.write(dir+'/'+filename+'_momtwo.fits', format='fits', overwrite=True)
    fits.writeto(dir+'/'+filename+'_momnine.fits', momnine.value, momtwo.header, overwrite=True)

def output_ascii_saa(self, outputdir):
    """
    Outputs an ascii table containing the information for each fit.
    """

    for i in range(len(self.wsaa)):
        saa_dict = self.saa_dict[i]
        if any(SAA.to_be_fit for SAA in saa_dict.values()):
            table = make_table(self, saa_dict, saa=True)
            table.write(outputdir+'/saa_model_solutions_'+str(self.wsaa[i])+'_.dat', format='ascii', \
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
                                      'chi2', 'dof', 'redchi2', 'aic' ]
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
    for j in range(len(saa_dict.keys())):
        # get the relavent SAA
        SAA = saa_dict[j]

        if SAA.to_be_fit:
            soln = SAA.model
            for k in range(int(soln.ncomps)):
                solution_desc = get_soln_desc(k, soln, \
                                              SAA.coordinates[1], \
                                              SAA.coordinates[0])
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
                                          spectrum.coordinates[1], \
                                          spectrum.coordinates[0])
            solnlist.append(solution_desc)
        else:
            for k in range(int(soln.ncomps)):
                solution_desc = get_soln_desc(k, soln, \
                                              spectrum.coordinates[1], \
                                              spectrum.coordinates[0])
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
                           soln.rms, soln.residstd, soln.chi2, \
                           soln.dof, soln.redchi2, soln.aic]

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
