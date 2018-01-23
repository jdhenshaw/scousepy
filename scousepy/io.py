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
import numpy as np
import os
import sys
import warnings
import shutil
import time

def mkdir_s1(outputdir, s1dir):
    """
    Make the output directory for stage 1
    """
    # TODO: error handling
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        os.mkdir(s1dir)
    else:
        shutil.rmtree(outputdir) #removes all the subdirectories!
        os.mkdir(outputdir)
        os.mkdir(s1dir)

def mkdir_s2(outputdir, s2dir):
    """
    Make the output directory for stage 2
    """
    if not os.path.exists(s2dir):
        os.mkdir(s2dir)
    else:
        # TODO: error handling
        pass

def mkdir_s3(outputdir, s3dir):
    """
    Make the output directory for stage 3
    """
    if not os.path.exists(s3dir):
        os.mkdir(s3dir)
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

    rsaa : len(N) list
           List of averaging radii

    fits_fmatter : a string formatter for output files to be written to
    """

    #for aver_cube in :
    hdu = fits.PrimaryHDU(data=saa_spectra, header=cube_header)
    hdu.header['RSAA'] = r
    hdu.writeto(dir+'/saa_cube_r_{}.fits'.format(r), overwrite=True)

def output_moments(momzero, momone, momtwo, dir, filename):
    """
    Write the moment maps to file
    """
    momzero.write(dir+'/'+filename+'_momzero.fits', format='fits', overwrite=True)
    momone.write(dir+'/'+filename+'_momone.fits', format='fits', overwrite=True)
    momtwo.write(dir+'/'+filename+'_momtwo.fits', format='fits', overwrite=True)

def output_ascii(self, outputdir):
    """
    Outputs an ascii table containing the information for each cluster.
    """

    for i in range(len(self.rsaa)):
        saa_dict = self.saa_dict[i]
        table = make_table(saa_dict)
        table.write(outputdir+'/saa_solutions_'+str(self.rsaa[i])+'_.dat', format='ascii', \
                    overwrite=True, delimiter='\t')
    return

def make_table(saa_dict, headings=None):
    """
    Generates an astropy table to hold the information
    """

    table = Table(meta={'name': 'best-fitting solutions'})

    headings=['ncomps', 'x', 'y', \
              'amplitude', 'err amplitude',
              'velocity', 'err velcoity',
              'FWHM', 'err FWHM', \
              'rms', 'residual', \
              'chi2', 'dof', 'redchi2', 'aic']

    solnlist = []
    for j in range(len(saa_dict.keys())):
        # get the relavent SAA
        SAA = saa_dict[j]

        if SAA.to_be_fit:
            list=[]
            soln = SAA.solution
            for k in range(int(soln.ncomps)):
                list = [soln.ncomps, SAA.coordinates[0], SAA.coordinates[1],\
                        soln.params[0+k*3], soln.errors[0+k*3],\
                        soln.params[1+k*3], soln.errors[1+k*3],\
                        soln.params[2+k*3], soln.errors[2+k*3],\
                        soln.rms, soln.residstd, \
                        soln.chi2, soln.dof, soln.redchi2, soln.aic]
                solnlist.append(list)

    solnarr = np.asarray(solnlist).T

    for j in range(len(solnarr[:,0])):
        table[headings[j]] = Column(solnarr[j,:])

    return table
    sys.exit()
