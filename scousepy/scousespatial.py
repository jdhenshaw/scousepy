# Licensed under an MIT open source license - see LICENSE

import numpy as np
import astropy.units as u
import os
import sys
import warnings
from astropy import wcs
from astropy import log
from .parallel_map import *
from tqdm import tqdm
warnings.simplefilter('ignore', wcs.FITSFixedWarning)

# add Python 2 xrange compatibility, to be removed
# later when we switch to numpy loops
if sys.version_info.major >= 3:
    range = range
    proto=3
else:
    range = xrange
    proto=2

class ScouseSpatial(object):
    """
    Code for identifying (and refitting) "bad" fits in ScousePy data

    Parameters
    ----------
    scouseobject : Instance of the scousepy class
    blocksize : int
        size of region used to define neighbours (default=5)
    flag_sigma : int
        compares measured values against measurement uncertainties, flags if
        measurement/uncertainty < flag_sigma (default=2)
    flag_deltancomps : int
        flag if the number of components in the reference pixel is greater than
        that of the weighted median of neighbours by more than flag_deltancomps
        (default=1)
    flag_ncomponentjump : int
        combines with flag_njumps. Establishes if the number of components in
        the reference pixel differs to that of neighbours by more than
        flag_ncomponentjump (default=2)
    flag_njumps : int
        flag if number of instances where number of components in
        the reference pixel differs to that of neighbours by more than
        flag_ncomponentjump is greater than flag_njumps (default=1)
    flag_nstddev : int
        relative difference in component properties as compared to mean neighbour
        model components (default=3)
    njobs : int
        number of threads for parallel processing
    verbose : bool
        shouty shouty
    """

    def __init__(self, scouseobject, blocksize=5,
                 flag_sigma=2, flag_deltancomps=1, flag_ncomponentjump=2,
                 flag_njumps=1, flag_nstddev=3, njobs=3, verbose=True):

        self.blocksize=blocksize
        self.xpos=None
        self.ypos=None
        self.flag_sigma=flag_sigma
        self.flag_deltancomps=flag_deltancomps
        self.flag_ncomponentjump=flag_ncomponentjump
        self.flag_njumps=flag_njumps
        self.flag_nstddev=flag_nstddev
        self.flag_dict={}
        from scousepy import scouse
        # load the cube
        fitsfile = os.path.join(scouseobject.datadirectory, scouseobject.filename+'.fits')
        scouse.load_cube(scouseobject, fitsfile=fitsfile)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_log = log.level
            log.setLevel('ERROR')
            self.cubeshape=scouseobject.cube.shape
            self.header=scouseobject.cube[0,:,:].header
            log.setLevel(old_log)
        self.flagmap=None
        self.outputdirectory=scouseobject.outputdirectory
        self.filename=scouseobject.filename
        self.spectral_axis=scouseobject.xtrim
        self.trimids=scouseobject.trimids
        self.tol = None
        self.njobs = njobs
        self.res=np.abs(np.diff(self.spectral_axis))[0]
        self.verbose=verbose
        self.fittype=scouseobject.fittype

    def flagging(self, indiv_dict, spectrum=None):
        """
        Method for flagging the data

        Parameters
        ----------
        indiv_dict : dictionary
            dictionary containing best-fitting solutions of a scousepy
            decomposition
        spectrum : instance of scousepy's spectrum class

        """
        if spectrum is None:
            spectrumlist=[spectrum for index, spectrum in indiv_dict.items()]
        else:
            spectrumlist=[spectrum]

        flagobjectlist = [self, indiv_dict]
        inputlist = [flagobjectlist+[spectrum] for spectrum in spectrumlist]

        # if njobs > 1 run in parallel else in series
        if self.njobs > 1:
            results=parallel_map(flagging_method, inputlist, numcores=self.njobs, verbose=self.verbose)
        else:
            if self.verbose:
                results=[flagging_method(input) for input in tqdm(inputlist)]
            else:
                results=[flagging_method(input) for input in inputlist]

        for result in results:
            self.flag_dict[result[0]]={'flag':result[1], 'compflag': result[2], 'paramflag': result[3]}

    def create_flag_map(self, indiv_dict, save_map=True, outputfits='flagmap.fits'):
        """
        Creates a map of the flagged pixels

        Parameters
        ----------
        indiv_dict : dictionary
            dictionary containing best-fitting solutions of a scousepy
            decomposition
        save_map : bool
            whether or not you would like to output a fits file
        outputfits : string
            name of the fits file to be produced

        """
        self.flagmap=np.zeros(self.cubeshape[1:])*np.nan
        for key,spectrum in indiv_dict.items():
            cx,cy = spectrum.coordinates
            if key in self.flag_dict:
                if np.any(self.flag_dict[key]['flag']):
                    index = np.where(np.asarray(self.flag_dict[key]['flag']))[0]
                    if 7 in index:
                        self.flagmap[cy, cx]=7
                    else:
                        self.flagmap[cy, cx]=index[0]
                else:
                    self.flagmap[cy, cx]=0
            else:
                self.flagmap[cy, cx]=0

        if save_map:
            self.save_map_to_fits(outputfits)

    def save_map_to_fits(self, outputfits):
        """
        Method to save the flag map to a fits file

        Parameters
        ----------
        outputfits : string
            name of the fits file to be produced
        """
        from astropy.io import fits
        savedir=self.outputdirectory+self.filename+'/stage_4/'

        fh = fits.PrimaryHDU(data=self.flagmap, header=self.header)
        fh.writeto(os.path.join(savedir, outputfits), overwrite=True)

    def spatial_refit(self, scouseobject, refitfile=None, tol=None):
        """
        Method for spatial refitting. Refitting is handled in a while loop which
        continuously modifies neighbour models as fitting is improved. Continues
        to loop until no more satisfactory fits can be identified. satisfactory
        here means fits which satisfy the flagging criteria

        Parameters
        ----------
        scouseobject : instance of the scousepy class

        """
        from .model_housing import individual_spectrum
        from copy import deepcopy
        import time

        if tol is None:
            self.tol=scouseobject.tol
        else:
            self.tol=tol

        indiv_dict = scouseobject.indiv_dict
        # create a list of flagged spectra
        flag = [spectrum for key, spectrum in indiv_dict.items() if key in self.flag_dict and np.any(self.flag_dict[key]['flag'])]
        lenflag = np.size(flag)
        # remove guesses from flagged spectra
        [setattr(spectrum,'guesses_updated',None) for key, spectrum in indiv_dict.items()]
        # stopping criteria
        stop=False

        while stop==False:
            # sort the list of flagged spectra according to the number of flag-free
            # neighbours
            sortedflag, sortednumneighbours = self.sortflag(flag, indiv_dict)
            # prepare the newflag list with spectra whose neighbours are all flagged
            newflag = [spectrum for key, spectrum in enumerate(sortedflag) if sortednumneighbours[key]==0.0]
            # remove spectra where all neighbours are flagged from the sortedflag list
            # this is our working list
            sortedflag = [spectrum for key, spectrum in enumerate(sortedflag) if sortednumneighbours[key]!=0.0]

            # method for checking model bank and refitting
            fittinglist=[self, indiv_dict]
            inputlist=[fittinglist+[spectrum] for spectrum in sortedflag]
            # if njobs > 1 run in parallel else in series
            if self.njobs > 1:
                fitresults = parallel_map(decomposition_method, inputlist, numcores=self.njobs, verbose=self.verbose)
            else:
                if self.verbose:
                    fitresults=[decomposition_method(input) for input in tqdm(inputlist)]
                else:
                    fitresults=[decomposition_method(input) for input in inputlist]

            # now add model solutions to the relevant spectra and add remaining
            # spectra to an output list
            count=0.0
            for i, fitresult in enumerate(fitresults):
                model=fitresult[0]
                guesses_updated=fitresult[1]
                spectrum=sortedflag[i]

                # any spectra for which an alternative solution cannot be
                # determined are appended to newflag
                if model is not None:
                    # deep copy spectrum
                    spectrum_=deepcopy(spectrum)
                    setattr(spectrum_, 'model', model)
                    # check if flagging satisfied
                    inputlist = [self, indiv_dict, spectrum_]
                    results=flagging_method(inputlist)

                    if np.all(np.invert(results[1])):
                        if spectrum.model not in spectrum.model_from_parent:
                            spectrum.model_from_parent.append(spectrum.model)

                        setattr(spectrum, 'model', model)
                        if model.method=='spatial':
                            individual_spectrum.add_model(spectrum, model)
                        self.flag_dict[results[0]]={'flag':results[1], 'compflag': results[2], 'paramflag': results[3]}
                    else:
                        # update the guesses and pass it back through
                        setattr(spectrum,'guesses_updated',guesses_updated,)
                        newflag.append(spectrum)
                else:
                    # update the guesses and pass it back through
                    setattr(spectrum,'guesses_updated',guesses_updated,)
                    newflag.append(spectrum)

            # stopping criteria - will break out of the loop when the length of
            # the flag and newflag lists are the same - indicating no more
            # suitable fits have been identified
            if len(newflag)==lenflag:
                stop=True
            else:
                flag = newflag
                lenflag=len(flag)

        # Save the scouse object automatically
        if scouseobject.autosave:
            import pickle
            if refitfile is not None:
                if os.path.exists(scouseobject.outputdirectory+scouseobject.filename+'/stage_4/'+refitfile):
                    os.rename(scouseobject.outputdirectory+scouseobject.filename+'/stage_4/'+refitfile,scouseobject.outputdirectory+scouseobject.filename+'/stage_4/'+refitfile+'.bk')

                with open(scouseobject.outputdirectory+scouseobject.filename+'/stage_4/'+refitfile, 'wb') as fh:
                    pickle.dump((scouseobject.completed_stages,scouseobject.check_spec_indices,indiv_dict), fh, protocol=proto)
            else:
                if os.path.exists(scouseobject.outputdirectory+scouseobject.filename+'/stage_4/s4.refit.scousepy'):
                    os.rename(scouseobject.outputdirectory+scouseobject.filename+'/stage_4/s4.scousepy',scouseobject.outputdirectory+scouseobject.filename+'/stage_4/s4.refit.scousepy.bk')

                with open(scouseobject.outputdirectory+scouseobject.filename+'/stage_4/s4.refit.scousepy', 'wb') as fh:
                    pickle.dump((scouseobject.completed_stages,scouseobject.check_spec_indices, indiv_dict), fh, protocol=proto)


    def sortflag(self, flag, indiv_dict):
        """
        Sorts list of flagged pixels according to the number of non-flagged
        neighbours. when this number is higher, there is a greater number of
        pixels from which to draw the mean neighbouring spectrum (i.e. higher
        accuracy).

        Parameters
        ----------
        flag : list
            list of flagged spectra
        indiv_dict : dictionary
            dictionary containing best-fitting solutions of a scousepy
            decomposition

        """
        numneighbours=[]
        flag = np.asarray(flag)
        for index, spectrum in enumerate(flag):
            # get the non-flagged neighbours
            nfneighbours = self.get_non_flagged_neighbours(spectrum, indiv_dict)
            numneighbours.append(np.size(nfneighbours))

        numneighbours=np.asarray(numneighbours)
        sortedids = np.argsort(numneighbours)[::-1]
        sortedflag, sortednumneighbours=flag[sortedids], numneighbours[sortedids]

        return sortedflag, sortednumneighbours

    def get_non_flagged_neighbours(self, spectrum, indiv_dict):
        """
        Method to identify the non-flagged neighbours surrounding a reference
        pixel

        Parameters
        ----------
        spectrum : instance of scousepy's spectrum class
        indiv_dict : dictionary
            dictionary containing best-fitting solutions of a scousepy
            decomposition
        """
        # get the coordinates of the spectrum
        xpos=spectrum.coordinates[0]
        ypos=spectrum.coordinates[1]
        # get the neighbours
        keys=self.get_neighbours(xpos, ypos)

        # get a list of neighbour spectra that they themselves are not flagged
        nfneighbours=[indiv_dict[key] for key in keys if key in indiv_dict.keys() and key in self.flag_dict and key is not np.nan and np.invert(np.any(self.flag_dict[key]['flag']))]
        # check to see if the spectrum is in our neighbour list and remove it
        if spectrum.index in [nfneighbour.index for nfneighbour in nfneighbours]:
            idspec=np.where([(spectrum.index == nfneighbour.index) for nfneighbour in nfneighbours])[0]
            del nfneighbours[idspec[0]]

        return nfneighbours

    def check_model_bank(self, spectrum, wmedian_ncomps, indiv_dict):
        """
        Method used to search within scousepy's model bank to find alternative
        solutions that satisfy the flagging criteria (overrides scousepy's
        default best-fit solution criteria which uses the AIC value)

        spectrum : instance of scousepy's spectrum class
        wmedian_ncomps : int
            the weighted median number of components in the neighbouring pixels
        indiv_dict : dictionary
            dictionary containing best-fitting solutions of a scousepy
            decomposition

        """
        from copy import deepcopy
        flags=[]
        ncomps=[]

        # first check to see if the current model is in the model bank and
        # remove
        alternative_models=deepcopy(spectrum.model_from_parent)
        if spectrum.model in alternative_models:
            alternative_models.remove(spectrum.model)

        if None in alternative_models:
            alternative_models.remove(None)

        if np.size(alternative_models)==0:
            flags=[False]
            model=None
        else:
            # first loop over the models to see if any of them satisfy the flag
            # criertia
            for i, altmodel in enumerate(alternative_models):
                # get the number of components
                ncomps.append(altmodel.ncomps)
                # create a copy of the spectrum
                spectrum_=deepcopy(spectrum)
                # update the selected model with the new model
                setattr(spectrum_, 'model', altmodel)
                # run flagging
                inputlist = [self, indiv_dict, spectrum_]
                results=flagging_method(inputlist)

                # check to see if the criteria is satisfied
                if np.any(results[1]):
                    flags.append(False)
                else:
                    flags.append(True)

            # if there are any cases where a suitable replacement has been found
            # we are going to pass that model back
            if np.any(flags):
                # check to see if any of the suitable models have the same number of
                # components as the median ncomps from the neighbours
                id = np.where((np.asarray(ncomps)==wmedian_ncomps) & (flags==True))[0]
                if np.size(id)!=0:
                    # if that is true select the first one that satisfies the condition
                    # (there could be more than one)
                    model=alternative_models[id[0]]
                else:
                    # if that is not true then we are going to still swap the model.
                    # loop over the possible models
                    for i, altmodel in enumerate(alternative_models):
                        if flags[i]:
                            model=altmodel
            else:
                # if no models are found return None
                model=None

        return model

    def save_flags(self, outputfile='s4.flags.scousepy'):
        """
        Method to save the flagging information. Outputs an instance of the
        ScouseSpatial class

        Parameters
        ----------
        outputfile : string
            file name
        """
        import pickle
        if os.path.exists(self.outputdirectory+self.filename+'/stage_4/'+outputfile):
            os.rename(self.outputdirectory+self.filename+'/stage_4/'+outputfile,self.outputdirectory+self.filename+'/stage_4/'+outputfile+'.bk')

        with open(self.outputdirectory+self.filename+'/stage_4/'+outputfile, 'wb') as fh:
            pickle.dump((self), fh, protocol=proto)

    @staticmethod
    def load_flags(fn):
        """
        Method to load in pre-flagged data

        Parameters
        ----------
        fn : string
            file name
        """
        import pickle
        with open(fn, 'rb') as fh:
            self = pickle.load(fh)

        return self

    def stats(self):
        """
        Method to produce some basic statistics on the flagged data

        """
        statkeys=['flaggedpixels', 'nflags','zerocomp','noneighbours','sigmaamp', 'sigmadisp', 'deltancomps','ncompjumps','sigmadiff']

        flaggedpixels=np.count_nonzero([np.any(self.flag_dict[key]['flag']) for key in self.flag_dict.keys()])
        nflags=np.sum([np.size(np.where(np.asarray(self.flag_dict[key]['flag']))[0]) for key in self.flag_dict.keys()])
        zerocomp=np.count_nonzero([(self.flag_dict[key]['flag'][1]) for key in self.flag_dict.keys()])
        noneighbours=np.count_nonzero([(self.flag_dict[key]['flag'][2]) for key in self.flag_dict.keys()])
        sigmaamp=np.count_nonzero([(self.flag_dict[key]['flag'][3]) for key in self.flag_dict.keys()])
        sigmadisp=np.count_nonzero([(self.flag_dict[key]['flag'][4]) for key in self.flag_dict.keys()])
        deltancomps=np.count_nonzero([(self.flag_dict[key]['flag'][5]) for key in self.flag_dict.keys()])
        ncompjumps=np.count_nonzero([(self.flag_dict[key]['flag'][6]) for key in self.flag_dict.keys()])
        sigmadiff=np.count_nonzero([(self.flag_dict[key]['flag'][7]) for key in self.flag_dict.keys()])

        flags=[flaggedpixels,nflags,zerocomp,noneighbours,sigmaamp,sigmadisp,deltancomps,ncompjumps,sigmadiff]
        statdict={}
        for i, key in enumerate(statkeys):
            statdict[key]=flags[i]

        return statdict

    def get_neighbours(self, xpos, ypos):
        """
        Returns a list of flattened indices for a given spectrum and its neighbours

        Parameters
        ----------
        xpos : int
            x position of the reference pixel
        ypos : int
            y position of the reference pixel
        """
        shape=self.cubeshape[1:]
        neighboursx=np.arange(xpos-(self.blocksize-1)/2,(xpos+(self.blocksize-1)/2)+1,dtype='int' )
        neighboursx=[x if (x>=0) & (x<=shape[1]-1) else np.nan for x in neighboursx ]
        neighboursy=np.arange(ypos-(self.blocksize-1)/2,(ypos+(self.blocksize-1)/2)+1,dtype='int' )
        neighboursy=[y if (y>=0) & (y<=shape[0]-1) else np.nan for y in neighboursy ]
        keys=[np.ravel_multi_index([y,x], shape)  if np.all(np.isfinite(np.asarray([y,x]))) else np.nan for y in neighboursy for x in neighboursx]

        return keys

    def get_weights(self, spectrum, neighbours):
        """
        Method to obtain 2D spatial weights for extracting mean neighbour model
        components. Uses a 2D Gaussian kernel

        Parameters
        ----------
        Spectrum : instance of scousepy's spectrum class
        neighbours : list
            A list of the neighbouring spectra

        """
        # get the central coordinate
        coord=spectrum.coordinates
        # get the coordinates of the neighbours
        neighbour_coords=np.asarray([neighbour.coordinates for neighbour in neighbours])
        # compute the distances between the central pixel and the neighbours
        distances=[np.linalg.norm(coord-neighbour_coord) for neighbour_coord in neighbour_coords]

        # create a gaussian kernel of sigma=1 and size blocksize
        from astropy.convolution import Gaussian2DKernel
        gaussian_2D_kernel = Gaussian2DKernel(1, x_size=self.blocksize, y_size=self.blocksize)
        # get the kernel values and normalise to the second highest value (highest is central pixel)
        kernel_values=gaussian_2D_kernel.array/sorted(gaussian_2D_kernel.array.flatten())[-2]

        # kernel centre coord
        kernel_centre=np.asarray([self.blocksize//2, self.blocksize//2])
        # get the kernel coords
        kernel_coords=np.transpose(np.where(gaussian_2D_kernel.array))
        # calculate the distances between the kernel centre and all other pixels in the kernel
        kernel_distances=[np.linalg.norm(kernel_centre-kernel_coord) for kernel_coord in kernel_coords]

        # get the weights associated with each coordinate
        kernel_weights=[kernel_values[kernel_coords[i,0], kernel_coords[i,1]] for i in range(len(kernel_coords[:,0]))]

        # tag the kernel weights to the distances of the neighbours in our data
        weights=[kernel_weights[id] for id in np.asarray([np.where(kernel_distances==distance)[0][0] for distance in distances])]

        return weights

    def weighted_median(self, data, weights):
        """
        Computes a weighted median based on input data and weights

        https://gist.github.com/tinybike/d9ff1dad515b66cc0d87

        Parameters
        ----------
        data : arr
            number of components in neighbouring pixels
        weights : arr
            spatial weights of the neighbours

        """
        data=np.asarray(data)
        weights=np.asarray(weights)

        sorted_data, sorted_weights = map(np.array, zip(*sorted(zip(data, weights))))

        midpoint = 0.5 * sum(sorted_weights)

        if any(weights > midpoint):
            idx=np.where(weights == np.max(weights))[0]
            weighted_median = (data[idx[0]])
        else:
            cs_weights = np.cumsum(sorted_weights)
            idx = np.where(cs_weights <= midpoint)[0][-1]
            if cs_weights[idx] == midpoint:
                weighted_median = np.mean(sorted_data[idx:idx + 2])
            else:
                weighted_median = sorted_data[idx + 1]

        return int(weighted_median)

    def get_component_jumps(self, ncomps_spec, ncomps_neighbours):
        """
        Method to compute number of component jumps

        Parameters
        ----------
        ncomps_spec : int
            number of components in the reference pixel
        ncomps_neighbours : arr
            number of components in neighbouring pixels
        """
        ncomp_jumps=np.asarray([np.abs(ncomps_spec-ncomps_neighbour) for ncomps_neighbour in ncomps_neighbours])
        return np.count_nonzero(ncomp_jumps>self.flag_ncomponentjump)

    def get_nstddev_from_mean_old(self, model, mean, stddev):
        nstddev=[]
        for _ in range(model.ncomps):
            nstddev.append([])
        for j in range(model.ncomps):
            mean_model=mean[j]
            stddev_model=stddev[j]
            component = model.params[int((j*len(model.parnames))):(j+1)*len(model.parnames)]
            for k in range(len(model.parnames)):
                if stddev_model[k]==0.0:
                    stddev_model[k]=1e-5
                nstd=np.absolute(component[k]-mean_model[k])/stddev_model[k]
                nstddev[j].append(nstd)

        return np.asarray(nstddev)

    def get_nstddev_from_mean(self, model, mean_params, mean_stddev):
        """
        Method to determine how many standard deviations individual components
        within the reference pixel are from the closest matching components in
        the mean model derived from neighbouring pixels

        Parameters
        ----------
        model : dict
            model describing the best-fit solution of the reference pixel
        mean_params : arr
            mean model derived from neighbouring pixels
        mean_stddev : arr
            standard deviation of mean model components
        """
        # get some important parameters
        model_ncomps = model.ncomps
        model_params = model.params
        mean_ncomps = int(len(mean_params)/len(model.parnames))

        nstddev=[]
        # loop over the number of components in the model
        for k in range(model_ncomps):
            diff=[]
            # get the first component
            component = model_params[int((k*len(model.parnames))):(k+1)*len(model.parnames)]
            for l in range(mean_ncomps):
                # compare this component against all components in the mean model
                pdiff = 0.0
                # get the first component
                mean_component = mean_params[int((l*len(model.parnames))):(l+1)*len(model.parnames)]
                for m in range(len(model.parnames)):
                    # compute the squared difference between properties
                    pdiff+=(component[m] - mean_component[m])**2.
                diff.append(np.sqrt(pdiff))
            # find the closest matching component
            idmin = np.where(diff==np.min(diff))[0]
            # get the closest matching component
            mean_component = mean_params[int((idmin[0]*len(model.parnames))):(idmin[0]+1)*len(model.parnames)]
            # get the relevant stddev values
            mean_component_stddev = mean_stddev[int((idmin[0]*len(model.parnames))):(idmin[0]+1)*len(model.parnames)]
            mean_component_stddev = [std if std !=0 else 1e-5 for std in mean_component_stddev]
            # compute number of standard deviations from the mean
            nstd=[np.absolute(component[k]-mean_component[k])/mean_component_stddev[k] for k in range(len(model.parnames))]
            nstddev.append(nstd)

        nstddev = np.asarray(nstddev)
        nstddev_av = np.asarray([np.sqrt((np.sum(nstddev[i,:]**2))/len(nstddev[0,:])) for i in range(len(nstddev[:,0]))])

        return nstddev,nstddev_av

    def get_mean_neighbour(self, model, neighbour_models, weights):
        """
        Method to determine the mean model derived from the pixels neighbouring
        a reference pixel

        Parameters
        ----------
        model : dict
            model describing the best-fit solution of the reference pixel
        neighbour_models : list
            list of neighbouring models
        weights : arr
            spatial weights of the neighbours
        """
        # first create the lists that we will populate
        neighbour_components=[]
        neighbour_weights=[]
        meanneighbour_params=[]
        meanneighbour_stddev=[]
        for _ in range(model.ncomps):
            neighbour_components.append([])
            neighbour_weights.append([])
            meanneighbour_params.append([])
            meanneighbour_stddev.append([])

        # find the neighbouring components
        for j in range(len(neighbour_models)):
            neighbour_components, neighbour_weights = self.find_closest_match(model, neighbour_models[j], neighbour_components, neighbour_weights, weights[j])

        # loop over the number of components
        for k in range(model.ncomps):
            if np.size(neighbour_components[k])!=0.0:
                for m in range(np.shape(neighbour_components[k])[1]):
                    property=np.asarray([neighbour_components[k][l][m] for l in range(np.shape(neighbour_components[k])[0])])
                    weighting=np.asarray(neighbour_weights[k])

                    idfinite=np.where(np.isfinite(property))
                    property=property[idfinite]
                    weighting=weighting[idfinite]

                    # compute the weighted averages
                    weighted_average=np.average(property, weights=neighbour_weights[k])
                    variance = np.average((np.asarray(property)-weighted_average)**2, weights=neighbour_weights[k])
                    meanneighbour_params[k].append(weighted_average)
                    meanneighbour_stddev[k].append(np.sqrt(variance))
            else:
                for m in range(len(model.parnames)):
                    meanneighbour_params[k].append(0.0)
                    meanneighbour_stddev[k].append(1e-10)

        meanneighbour_params, meanneighbour_stddev = np.asarray(meanneighbour_params), np.asarray(meanneighbour_stddev)

        meanneighbour_params = meanneighbour_params.flatten()
        meanneighbour_stddev = meanneighbour_stddev.flatten()

        idx=[(param != 0.0) for param in meanneighbour_params]

        meanneighbour_params = meanneighbour_params[idx]
        meanneighbour_stddev = meanneighbour_stddev[idx]

        return  meanneighbour_params, meanneighbour_stddev

    def get_mean_neighbour_zerocomps(self, spectrum, neighbours, wmedian_ncomps):
        """
        Get the mean neighbour model for reference pixels that have no
        best-fitting solution

        Parameters
        ----------
        Spectrum : instance of scousepy's spectrum class
        Neighbours : arr
            arr containing neighbouring spectra
        wmedian_ncomps : int
            the weighted median number of components in the neighbouring pixels
        """
        # get all the neighbours that have wmedian_ncomps number of comps
        neighbours = [neighbour for neighbour in neighbours if (neighbour.model.ncomps==wmedian_ncomps)]
        # get the weights
        weights_wmedian_ncomps = self.get_weights(spectrum, neighbours)
        # get the weighted mean spectrum
        neighbourparams=np.asarray([neighbour.model.params for neighbour in neighbours])
        neighbourerrs=np.asarray([neighbour.model.errors for neighbour in neighbours])
        # reshape the arrays
        neighbourparams=np.reshape(neighbourparams, (np.shape(neighbourparams)[0], wmedian_ncomps, len(spectrum.model.parnames)))
        neighbourerrs=np.reshape(neighbourerrs, (np.shape(neighbourerrs)[0], wmedian_ncomps, len(spectrum.model.parnames)))

        # calculate weighted averages
        meanneighbour_params=np.average(neighbourparams, axis=0, weights=weights_wmedian_ncomps)
        meanneighbour_var=np.average((neighbourparams-meanneighbour_params)**2, axis=0, weights=weights_wmedian_ncomps)
        meanneighbour_stddev=np.sqrt(meanneighbour_var)

        meanneighbour_params, meanneighbour_stddev = np.asarray(meanneighbour_params), np.asarray(meanneighbour_stddev)

        meanneighbour_params = meanneighbour_params.flatten()
        meanneighbour_stddev = meanneighbour_stddev.flatten()

        idx=[(param != 0.0) for param in meanneighbour_params]

        meanneighbour_params = meanneighbour_params[idx]
        meanneighbour_stddev = meanneighbour_stddev[idx]

        return  meanneighbour_params, meanneighbour_stddev

    def find_closest_match(self,model,model_neighbour,neighbour_components,neighbour_weights,weight):
        """
        Finds the closest match between a reference pixel component and those in
        the mean model derived from neighbours

        Parameters
        ----------
        model : dict
            model describing the best-fit solution of the reference pixel
        model_neighbour
            model describing the best-fit solution of the neighbouring pixels
        neighbour_components : list
            list to be updated with the closest matching neighbour components
        neighbour_weights : arr
            list to be updated with the weight information
        weight : arr
            weight values for neighbouring pixels

        """

        diff=[]

        for k in range(model.ncomps):
            component = model.params[int((k*len(model.parnames))):(k+1)*len(model.parnames)]
            for l in range(model_neighbour.ncomps):
                pdiff = 0.0
                component_neighbour = model_neighbour.params[int((l*len(model.parnames))):(l+1)*len(model.parnames)]
                for m in range(len(model.parnames)):
                    pdiff+=(component[m] - component_neighbour[m])**2.
                diff.append(np.sqrt(pdiff))

        diff=np.asarray(diff)

        for k in range(model_neighbour.ncomps):
            idmin = np.where(diff[int((k*model.ncomps)):int(((k+1)*model.ncomps))] == np.min(diff[int((k*model.ncomps)):int(((k+1)*model.ncomps))]))[0]
            idmin = idmin[0]
            component_neighbour = model_neighbour.params[int((k*len(model_neighbour.parnames))):(k+1)*len(model_neighbour.parnames)]
            neighbour_components[idmin].append(component_neighbour)
            if model_neighbour.ncomps == model.ncomps:
                neighbour_weights[idmin].append(weight)
            else:
                neighbour_weights[idmin].append(weight**2.)

        return neighbour_components, neighbour_weights

    def check_resolved(self, spectrum):
        """
        Checks to see if all components in the reference spectrum model are
        resolved

        Parameters
        ----------
        spectrum : instance of scousepy's spectrum class

        """

        fwhmconv = 2.*np.sqrt(2.*np.log(2.))

        # Find where the velocity dispersion is located in the parameter array
        namelist = ['dispersion', 'width', 'fwhm']
        foundname = [pname in namelist for pname in spectrum.model.parnames]
        foundname = np.array(foundname)
        idx=np.where(foundname==True)[0]
        idx=np.asscalar(idx[0])

        nparams=np.size(spectrum.model.parnames)
        ncomponents=spectrum.model.ncomps

        disparr = np.asarray([spectrum.model.params[int((i*nparams)+idx)] for i in range(ncomponents)])

        if np.any(disparr*fwhmconv < self.res):
            flag=True
        else:
            flag=False

        return flag

    def check_amplitude(self, spectrum):
        """
        Checks to see if the amplitude of each component is > flag_sigma *
        measurement uncertainty

        Parameters
        ----------
        spectrum : instance of scousepy's spectrum class

        """
        # Find where in the parameter array the "amplitude" is located. Make this
        # general to allow for other models
        namelist = ['tex', 'amp', 'amplitude', 'peak', 'tant', 'tmb']
        foundname = [pname in namelist for pname in spectrum.model.parnames]
        foundname = np.array(foundname)
        idx=np.where(foundname==True)[0]
        idx=np.asscalar(idx[0])

        nparams=np.size(spectrum.model.parnames)
        ncomponents=spectrum.model.ncomps

        amparr = np.asarray([spectrum.model.params[int((i*nparams)+idx)] for i in range(ncomponents)])
        erramparr = np.asarray([spectrum.model.errors[int((i*nparams)+idx)] for i in range(ncomponents)])

        sigmaarr = amparr/erramparr
        if np.any(sigmaarr < self.flag_sigma):
            flag=True
        else:
            flag=False

        return flag

    def check_sigma(self, spectrum):
        """
        Checks to see if the dispersion of each component is > flag_sigma *
        measurement uncertainty

        Parameters
        ----------
        spectrum : instance of scousepy's spectrum class

        """
        # Find where the velocity dispersion is located in the parameter array
        namelist = ['dispersion', 'width', 'fwhm']
        foundname = [pname in namelist for pname in spectrum.model.parnames]
        foundname = np.array(foundname)
        idx=np.where(foundname==True)[0]
        idx=np.asscalar(idx[0])

        nparams=np.size(spectrum.model.parnames)
        ncomponents=spectrum.model.ncomps

        disparr = np.asarray([spectrum.model.params[int((i*nparams)+idx)] for i in range(ncomponents)])
        errdisparr = np.asarray([spectrum.model.errors[int((i*nparams)+idx)] for i in range(ncomponents)])

        sigmaarr = disparr/errdisparr
        if np.any(sigmaarr < self.flag_sigma):
            flag=True
        else:
            flag=False

        return flag

def flagging_method(input):
    """
    Method for flagging the data.

    Flags:
    1 : zero components
    2 : no neighbours
    3 : amplitude check
    4 : sigma check
    5 : deltancomps
    6 : ncomponent jumps
    7 : sigma diff

    Parameters
    ----------
    input : list
        list containing an instance of the ScouseSpatial class, the indiv_dict,
        and a spectrum.
    """

    self, indiv_dict, spectrum = input

    # get the coordinates of the spectrum
    xpos=spectrum.coordinates[0]
    ypos=spectrum.coordinates[1]
    # get the neighbours
    keys=self.get_neighbours(xpos,ypos)
    # get a list of neighbour spectra
    neighbours=[indiv_dict[key] for key in keys if key in indiv_dict.keys() and key is not np.nan and indiv_dict[key].model.ncomps != 0.0]
    # check to see if the spectrum is in our neighbour list and remove it
    if spectrum.index in [neighbour.index for neighbour in neighbours]:
        idspec=np.where([(spectrum.index == neighbour.index) for neighbour in neighbours])[0]
        del neighbours[idspec[0]]

    neighbours_ncomps=[neighbour.model.ncomps for neighbour in neighbours]
    # get weights
    weights=self.get_weights(spectrum, neighbours)

    # flags:
    flag=[False, False, False, False, False, False, False, False]

    # flag zero component models
    if spectrum.model.ncomps == 0.0:
        # note that the indices are 1-based for ease of plotting
        flag[1]=True
        compflag=[]
        paramflag=[]
    else:
        if np.size(neighbours)==0:
            flag[2]=True
            compflag=[]
            paramflag=[]
        else:
            # flag if measurement uncertainty*flag_sigma > amplitude
            flag[3]=self.check_amplitude(spectrum)

            # flag if measurement uncertainty*flag_sigma > sigma
            flag[4]=self.check_sigma(spectrum)

            # flag deviations of weighted median number of components
            wmedian_ncomps=self.weighted_median(neighbours_ncomps, weights)
            # compute delta ncomps
            delta_ncomps=np.abs(spectrum.model.ncomps-wmedian_ncomps)

            # flag if delta ncomps > self.flag_deltancomps - make user defined
            if delta_ncomps > self.flag_deltancomps:
                flag[5]=True

            # flag number of component jumps
            # determine the number of occurances where the difference in ncomps
            # between the spectrum and its neighbours is > self.flag_ncomponentjump
            ncomponent_jumps = self.get_component_jumps(spectrum.model.ncomps, neighbours_ncomps)
            # flag the pixel if the number of component jumps that are > self.flag_ncomponentjump
            # is > self.flag_njumps
            if ncomponent_jumps > self.flag_njumps:
                flag[6]=True

            # flag spectrum properties
            meanneighbour_params, meanneighbour_stddev = self.get_mean_neighbour(spectrum.model, [spec.model for spec in neighbours], weights)
            nstddev,nstddev_av = self.get_nstddev_from_mean(spectrum.model, meanneighbour_params, meanneighbour_stddev)

            if np.any(nstddev_av>self.flag_nstddev):
                flag[7]=True
                # which component is flagged
                compflag=[np.any(nstddev[j]>self.flag_nstddev) for j in range(spectrum.model.ncomps)]
                # which parameter is flagged
                paramflag=[(nstddev[j][k]>self.flag_nstddev) for j in range(spectrum.model.ncomps) for k in range(len(spectrum.model.parnames))]
            else:
                compflag=[]
                paramflag=[]

    return [spectrum.index, flag, compflag, paramflag]

def decomposition_method(input):
    """
    Method used for refitting the data

    Parameters
    ----------
    input : list
        list containing an instance of the ScouseSpatial class, the indiv_dict,
        and a spectrum.
    """
    from .SpectralDecomposer import Decomposer
    from .model_housing import indivmodel
    # unpack the inputs
    self,indiv_dict,spectrum = input

    # get the non-flagged neighbours
    neighbours = self.get_non_flagged_neighbours(spectrum, indiv_dict)
    # get the number of components in the non-flagged neighbours
    neighbours_ncomps=[neighbour.model.ncomps for neighbour in neighbours]
    # get weights
    weights=self.get_weights(spectrum, neighbours)
    # get the weighted median number of ncomps
    wmedian_ncomps=self.weighted_median(neighbours_ncomps, weights)
    # get the mean neighbour properties
    if spectrum.model.ncomps!=0:
        meanneighbour_params, meanneighbour_stddev = self.get_mean_neighbour(spectrum.model, [spec.model for spec in neighbours], weights)
        model=self.check_model_bank(spectrum, wmedian_ncomps, indiv_dict)
        if model is not None:
            guesses_updated=None
    else:
        meanneighbour_params, meanneighbour_stddev = self.get_mean_neighbour_zerocomps(spectrum, neighbours, wmedian_ncomps)
        model=None

    if model is None:
        if len(meanneighbour_params)!=0:
            # set up the decomposer
            decomposer=Decomposer(self.spectral_axis,spectrum.spectrum[self.trimids],spectrum.rms)
            setattr(decomposer,'psktemplate',spectrum.template,)

            # inputs to initiate the fitter
            if np.size(spectrum.guesses_updated)<=1:
                guesses=meanneighbour_params
            else:
                guesses=spectrum.guesses_updated

            # always pass the parent SAA parameters for comparison
            guesses_parent=meanneighbour_params
            # fit the spectrum
            Decomposer.fit_spectrum_from_parent(decomposer,guesses,guesses_parent,self.tol,self.res,fittype=self.fittype,method='spatial')

            # # generate a model
            if decomposer.validfit:
                model=indivmodel(decomposer.modeldict)
                guesses_updated=decomposer.guesses_updated
            else:
                model=None
                guesses_updated=decomposer.guesses_updated
        else:
            model=None
            guesses_updated=None

    return [model, guesses_updated]
