import numpy as np
from astropy.io import fits
from astropy import units as u
import sys
from astropy import wcs
import pyspeckit

def get_xaxis(header):
    """
    Generate & return the velocity axis from the fits header.
    """
    mywcs = wcs.WCS(header)
    specwcs = mywcs.sub([wcs.WCSSUB_SPECTRAL])
    return np.squeeze(specwcs.wcs_pix2world(np.arange(header['NAXIS{0}'.format(mywcs.wcs.spec+1)]), 0))

def get_spec(self, x, y):
    """
    Generate the spectrum
    """
    return pyspeckit.Spectrum(data=y, error=np.ones(1000)*0.01, xarr=x, doplot=True, unit=self.cube.header['BUNIT'])
