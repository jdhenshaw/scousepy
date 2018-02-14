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
    rsaa             =  [4.0]
    # Enter an approximate rms value for the data.
    rms_approx       =  0.1
    # Threshold below which all channel values set to 0.0
    sigma_cut        =  3.0
    # Tolerances for stage_3
    tol              = [3.0, 2.0, 2.5, 2.5, 0.5]
    # Spectral resolution
    specres          = 0.07

    RG = True
    nRG = 2.
    TS = False
    verb = True
    fittype = 'gaussian'
    njobs = 4

    #s = scouse.stage_1(filename, datadirectory, ppv_vol, rsaa, rms_approx, sigma_cut, verbose = verb, training_set=TS, samplesize=1, write_moments=True, save_fig=True)
    #s = scouse.stage_1(filename, datadirectory, ppv_vol, rsaa, rms_approx, sigma_cut, fittype=fittype, verbose = verb, training_set=TS, samplesize=1, refine_grid=RG, nrefine = nRG, write_moments=True, save_fig=True)
    #s = scouse.load_from(datadirectory+filename+'/stage_2/s2.scousepy')
    #s = scouse.stage_2(s, verbose=verb, write_ascii=True)
    #s = scouse.load_from(datadirectory+filename+'/stage_2/s2.scousepy')
    #s = scouse.stage_3(s, tol, njobs=njobs, verbose=verb)
    #s = scouse.stage_4(s, verbose=verb)
    #s = scouse.load_from(datadirectory+filename+'/stage_4/s4.scousepy')
    #s = scouse.stage_5(s, blocksize = 6, figsize = [18,10], plot_residuals=True, verbose=verb, blockrange=[0,50])
    s = scouse.load_from(datadirectory+filename+'/stage_6/s6.scousepy')
    s = scouse.stage_6(s, plot_neighbours=True, radius_pix = 2, figsize = [18,10], plot_residuals=True, write_ascii=True, verbose=verb, specrange=[1,2])

run_scousepy()
