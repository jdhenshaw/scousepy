from scousepy import scouse
from astropy.io import fits
import os

def run_scousepy():
    # Input values for core SCOUSE stages
    datadirectory    =  './'
    # The data cube to be analysed
    filename         =  'n2h+10_37'
    # The range in velocity, x, and y over which to fit
    ppv_vol          =  [32.0,42.0,0.0,0.0,0.0,0.0]
    # Radius for the spectral averaging areas. Pixel units.
    rsaa             =  [2.0]
    # Enter an approximate rms value for the data.
    rms_approx       =  0.05
    # Threshold below which all channel values set to 0.0
    sigma_cut        =  3.0

    s = scouse.stage_1(filename, datadirectory, ppv_vol, rsaa, rms_approx, sigma_cut, verbose = True, training_set=True, samplesize=12, write_moments=True, save_fig=False)
    s = scouse.stage_2(s, verbose=True, training_set=True)

run_scousepy()
