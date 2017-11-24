from astropy import units as u
from astropy.io import fits
from spectral_cube import SpectralCube
from astropy import wcs

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
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        os.mkdir(s1dir)
    else:
        shutil.rmtree(outputdir) #removes all the subdirectories!
        os.mkdir(outputdir)
        os.mkdir(s1dir)

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
