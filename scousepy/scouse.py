"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2017 Jonathan D. Henshaw
CONTACT: j.d.henshaw[AT]ljmu.ac.uk

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
warnings.simplefilter('ignore', wcs.FITSFixedWarning)

from .stage_1 import *
from .io import *
from .progressbar import AnimatedProgressBar

# add Python 2 xrange compatibility, to be removed
# later when we switch to numpy loops
if sys.version_info.major >= 3:
    range = range
else:
    range = xrange

try:
    input = raw_input
except NameError:
    pass

class scouse(object):

    def __init__(self):

        self.data = None

    @staticmethod
    def stage_1(filename, datadirectory, ppv_vol, rsaa, rms_approx, sigma_cut, verbose = False, outputdir=None, write_moments=False, save_fig=True, training_set=False, samplesize=10):

        starttime = time.time()
        # Generate file structure
        if outputdir==None:
            outputdir=datadirectory

        # directory structure
        fitsfile = os.path.join(datadirectory, filename+'.fits')
        outputdirectory = os.path.join(outputdir, filename)
        s1dir = os.path.join(outputdir, filename, 'stage_1')

        # create new directory
        mkdir_s1(outputdirectory, s1dir)

        if verbose:
            progress_bar = print_to_terminal(stage='s1', step='start')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_log = log.level
            log.setLevel('ERROR')
            # Read in the datacube
            cube = SpectralCube.read(fitsfile).with_spectral_unit(u.km/u.s)
            # Generate moment maps
            momzero, momone, momtwo = get_moments(cube, ppv_vol, rms_approx, sigma_cut, write_moments, s1dir, filename, verbose)
            # get the coverage / average the subcube spectra
            coverage_coordinates, saa_spectra = [], []
            for r in rsaa:
                cc, ss = define_coverage(cube, momzero.value, r, verbose)
                if training_set:
                    cc, ss = get_random_saa(cc, ss, samplesize, rsaa, verbose)
                else:
                    if verbose:
                        print('')
                _ss=np.array(ss)
                write_averaged_spectra(cube.header, ss, r, s1dir)
                coverage_coordinates.append(cc)
                saa_spectra.append(ss)
            log.setLevel(old_log)

        if save_fig:
            # plot multiple coverage areas
            plot_rsaa(coverage_coordinates, momzero.value, rsaa, s1dir, filename)

        endtime = time.time()

        if verbose:
            progress_bar = print_to_terminal(stage='s1', step='end', length=np.size(momzero), var=cc, t1=starttime, t2=endtime)

    @staticmethod
    def stage_2():
        return
