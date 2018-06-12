from scousepy import scouse
from astropy.io import fits
import os
import sys
import pylab as pl
pl.ion()

def run_scousepy():
    # Input values for core SCOUSE stages (see help in scouse.py for more info)
    # Data directory - scouse will dump the output stuff into the same directory
    # as your fits file by default but you can change this using the keyword
    # outputdir.
    datadirectory    =  './'
    # The data cube to be analysed - remove .fits
    filename         =  'n2h+10_37'
    # The range in velocity, x, and y over which to fit. This is an optional
    # keyword - not setting it will mean the whole cube will be fit - we don't
    # want this here as we are fitting the isolated hyperfine of n2h+ (1-0)
    ppv_vol          =  [32.0,42.0,None,None,None,None]
    # Radius for the spectral averaging areas. Pixel units.
    wsaa             =  [8.0]
    # Tolerances for stage_3; see henshaw et al. 2016a for details. I'd keep the
    # first and the last the same then tweak the others as necessary.
    tol              = [3.0, 1.0, 3.0, 3.0, 0.5]

    refine_grid = True
    number_of_refinements = 2.
    verb = True
    fittype = 'gaussian'
    njobs = 1 # This is used for stage 3 mainly. Stages 1,2,4,5,6 will most
              # likely be fine on a laptop but for big cubes stage 3 may need
              # more oompf.
    mask = 0.3 # All data below this value are masked during moment analysis.

    #==========================================================================#
    # This is stage 1 - a description can be found on the github page - used to
    # define the coverage.
    #==========================================================================#
    if os.path.exists(datadirectory+filename+'/stage_1/s1.scousepy'):
        s = scouse(outputdir=datadirectory, filename=filename, fittype=fittype,
                   datadirectory=datadirectory)
        s.load_stage_1(datadirectory+filename+'/stage_1/s1.scousepy')
        s.load_cube(fitsfile=filename+".fits")
    else:
        s = scouse.stage_1(filename, datadirectory, wsaa, ppv_vol=ppv_vol, mask_below=mask, fittype=fittype, verbose = verb, refine_grid=refine_grid, nrefine = number_of_refinements, write_moments=True, save_fig=True)

    #==========================================================================#
    # Stage 2 - this is the manual fitting step.
    #==========================================================================#
    # if os.path.exists(datadirectory+filename+'/stage_2/s2.scousepy'):
    #     s.load_stage_2(datadirectory+filename+'/stage_2/s2.scousepy')
    # else:
    #     s = scouse.stage_2(s, verbose=verb, write_ascii=True)

    #==========================================================================#
    #stage 3 - automated fitting
    #==========================================================================#
    # if os.path.exists(datadirectory+filename+'/stage_3/s3.scousepy'):
    #     s.load_stage_3(datadirectory+filename+'/stage_3/s3.scousepy')
    # else:
    #     s = scouse.stage_3(s, tol, njobs=njobs, verbose=verb)

    #==========================================================================#
    # Stage 4 - selecting the best fits
    #==========================================================================#
    # if os.path.exists(datadirectory+filename+'/stage_4/s4.scousepy'):
    #     s.load_stage_4(datadirectory+filename+'/stage_4/s4.scousepy')
    # else:
    #     s = scouse.stage_4(s, verbose=verb)

    #==========================================================================#
    # stage 5 - checking the fits - this is interactive. It will show some
    # helpful diagnostic plots for you to identify where the fits aren't very
    # good. If you click the pixel you are interested in it will bring up a
    # area of "blocksize"^2 pixels. Select ones which look bad. Pressing enter
    # will return you to the previous plot. Pressing enter on that plot will end
    # stage 5.
    #==========================================================================#
    # if os.path.exists(datadirectory+filename+'/stage_5/s5.scousepy'):
    #     s.load_stage_5(datadirectory+filename+'/stage_5/s5.scousepy')
    # else:
    #     s = scouse.stage_5(s, blocksize = 6, figsize = [18,10], plot_residuals=True, verbose=verb)

    #==========================================================================#
    # Stage 6 - refitting the data you identified in stage 5
    #==========================================================================#
    # if os.path.exists(datadirectory+filename+'/stage_6/s6.scousepy'):
    #     s.load_stage_6(datadirectory+filename+'/stage_6/s6.scousepy')
    # else:
    #     s = scouse.stage_6(s, plot_neighbours=True, radius_pix = 2, figsize = [18,10], plot_residuals=True, write_ascii=True, verbose=verb)

    return s

s = run_scousepy()
