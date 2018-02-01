from scousepy import scouse
from astropy.io import fits
import os
import sys

def run_scousepy():
    # Input values for core SCOUSE stages
    datadirectory    =  './'
    # The data cube to be analysed
    filename         =  'n2h+10_37'
    # The range in velocity, x, and y over which to fit
    ppv_vol          =  [32.0,42.0,0.0,0.0,0.0,0.0]
    # Radius for the spectral averaging areas. Pixel units.
    rsaa             =  [20.0,10.0]
    # Enter an approximate rms value for the data.
    rms_approx       =  0.1
    # Threshold below which all channel values set to 0.0
    sigma_cut        =  3.0
    # Tolerances for stage_3
    tol              = [3.0, 2.0, 1.5, 1.5, 0.5]
    # Spectral resolution
    specres          = 1.7

    RG = False
    nRG = 3.
    TS = False
    verb = True

    #s = scouse.stage_1(filename, datadirectory, ppv_vol, rsaa, rms_approx, sigma_cut, verbose = verb, training_set=TS, samplesize=1, write_moments=True, save_fig=True)
    s = scouse.stage_1(filename, datadirectory, ppv_vol, rsaa, rms_approx, sigma_cut, verbose = verb, training_set=TS, samplesize=1, refine_grid=RG, nrefine = nRG, write_moments=True, save_fig=True)
    s = scouse.stage_2(s, verbose=verb, training_set=TS, write_ascii=True)
    s = scouse.stage_3(s, tol, verbose=verb, training_set=TS)

run_scousepy()
