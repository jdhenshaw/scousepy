import numpy as np
from scousepy import scouse
from spectral_cube import SpectralCube
from astropy.io import fits
from astropy.convolution import Box2DKernel
from astropy import units as u
import os
import sys

import pylab as pl
#pl.ioff()
pl.ion()

# Input values for core SCOUSE stages
datadirectory    =  './'
# The data cube to be analysed
filename         =  'n2h+10_37_ds'
# The range in velocity, x, and y over which to fit
ppv_vol          =  [32.0,42.0,0.0,0.0,0.0,0.0]
# Radius for the spectral averaging areas. Pixel units.
rsaa             =  [2.0]
# Tolerances for stage_3
# peak minimum S/N, minimum width in channels, maximum difference from SAA velocity, maximum difference from SAA FWHM, maximum separation in units FWHM
tol              = [3.0, 1.0, 3.0, 2.0, 0.5]
# Spectral resolution
#specres          = 0.07

RG = False
nRG = 1.
TS = False
verb = True
fittype = 'gaussian'
njobs = 1

cube = SpectralCube.read('n2h+10_37.fits').with_spectral_unit(u.km/u.s)
# smooth and downsample for first-round fitting
# (smoothing is maybe optional..)
dscube = cube.spatial_smooth(Box2DKernel(2))[:,::2,::2]
dscube.write('n2h+10_37_ds.fits', overwrite=True)

# run the full scouse suite on the downsampled data
if os.path.exists(datadirectory+filename+'/stage_1/s1.scousepy'):
    s = scouse(outputdir=datadirectory, filename=filename, fittype=fittype,
               datadirectory=datadirectory)
    s.load_stage_1(datadirectory+filename+'/stage_1/s1.scousepy')
    s.load_cube(fitsfile=filename+".fits")
else:
    s = scouse.stage_1(filename, datadirectory, ppv_vol, rsaa, mask_below=0.3, fittype=fittype, verbose = verb, refine_grid=RG, nrefine = nRG, write_moments=True, save_fig=True)

if os.path.exists(datadirectory+filename+'/stage_2/s2.scousepy'):
    s.load_stage_2(datadirectory+filename+'/stage_2/s2.scousepy')
else:
    s = scouse.stage_2(s, verbose=verb, write_ascii=True)

if os.path.exists(datadirectory+filename+'/stage_3/s3.scousepy'):
    s.load_stage_3(datadirectory+filename+'/stage_3/s3.scousepy')
else:
    s = scouse.stage_3(s, tol, njobs=njobs, verbose=verb)

if os.path.exists(datadirectory+filename+'/stage_4/s4.scousepy'):
    s.load_stage_4(datadirectory+filename+'/stage_4/s4.scousepy')
else:
    s = scouse.stage_4(s, verbose=verb)

if os.path.exists(datadirectory+filename+'/stage_5/s5.scousepy'):
    s.load_stage_5(datadirectory+filename+'/stage_5/s5.scousepy')
else:
    s = scouse.stage_5(s, blocksize = 6, figsize = [18,10], plot_residuals=True, verbose=verb)

if os.path.exists(datadirectory+filename+'/stage_6/s6.scousepy'):
    s.load_stage_6(datadirectory+filename+'/stage_6/s6.scousepy')
else:
    s = scouse.stage_6(s, plot_neighbours=True, radius_pix = 2, figsize = [18,10], plot_residuals=True, write_ascii=True, verbose=verb)

# now the downsampled data are done, so we'll expand the guesses to all pixels
# and do the automatic fitting steps

# first, we get the linear (flat) indices of each pixel in the full-resolution data
index_shape = np.arange(cube[0,:,:].size).reshape(cube.shape[1:])

# create Spectral Averaging Areas from each of the fitted spectra in the low-res data
from scousepy.saa_description import saa
new_saa = {ind: saa.from_indiv_spectrum(spec, scouse=s, sample=True)
           for ind,spec in s.indiv_dict.items()}
for ind,SAA in new_saa.items():
    # x,y coordinates were 1/2, so multiply them
    SAA._coordinates *= 2
    cy,cx = SAA.coordinates
    # fill in the flat indices.  _indices_flat is how the SAA and indiv spectra are associated
    SAA._indices_flat = [index_shape[y,x] for (y,x) in [(cy,cx), (cy+1, cx), (cy, cx+1), (cy+1, cx+1)]
                         if y<index_shape.shape[0] and x<index_shape.shape[1]]
    # populate the various models from the fitted dict
    SAA._model = s.indiv_dict[ind]._model
    SAA._models = s.indiv_dict[ind]._models
    SAA._model_dud = s.indiv_dict[ind]._model_dud
    SAA._model_parent = s.indiv_dict[ind]._model_parent
    SAA._model_spatial = s.indiv_dict[ind]._model_spatial

# change parameters back to the full-resolution version
filename         =  'n2h+10_37'
s.filename = filename
s.outputdirectory = './n2h+10_37'
s.cube = cube
s.saa_dict = {0: new_saa}
s.indiv_dict = {}

# run stages 3-6 with the high-res data
if os.path.exists(datadirectory+filename+'/stage_3/s3.scousepy'):
    s.load_stage_3(datadirectory+filename+'/stage_3/s3.scousepy')
else:
    s = scouse.stage_3(s, tol, njobs=njobs, verbose=verb)

if os.path.exists(datadirectory+filename+'/stage_4/s4.scousepy'):
    s.load_stage_4(datadirectory+filename+'/stage_4/s4.scousepy')
else:
    s = scouse.stage_4(s, verbose=verb)

if os.path.exists(datadirectory+filename+'/stage_5/s5.scousepy'):
    s.load_stage_5(datadirectory+filename+'/stage_5/s5.scousepy')
else:
    s = scouse.stage_5(s, blocksize = 6, figsize = [18,10], plot_residuals=True, verbose=verb)

if os.path.exists(datadirectory+filename+'/stage_6/s6.scousepy'):
    s.load_stage_6(datadirectory+filename+'/stage_6/s6.scousepy')
else:
    s = scouse.stage_6(s, plot_neighbours=True, radius_pix = 2, figsize = [18,10], plot_residuals=True, write_ascii=True, verbose=verb)
