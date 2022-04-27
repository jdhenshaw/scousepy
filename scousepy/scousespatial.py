# Licensed under an MIT open source license - see LICENSE

import numpy as np
import astropy.units as u
import os
import sys
import warnings
from astropy import wcs
from astropy import log
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
    def __init__(self, scouseobject, blocksize=5,
                 flag_deltancomps=1, flag_ncomponentjump=2,
                 flag_njumps=1, flag_nstddev=3):

        self.blocksize=blocksize
        self.xpos=None
        self.ypos=None
        self.flag_deltancomps=flag_deltancomps
        self.flag_ncomponentjump=flag_ncomponentjump
        self.flag_njumps=flag_njumps
        self.flag_nstddev=flag_nstddev
        self.indiv_dict=scouseobject.indiv_dict
        # for index, spectrum in self.indiv_dict.items():
        #     if hasattr(spectrum, 'spectrum'):
        #         delattr(spectrum, 'spectrum')
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
        self.tol = scouseobject.tol
        self.res=np.abs(np.diff(self.spectral_axis))[0]

    def flagging(self, scouseobject):
        """

        """
        for index, spectrum in self.indiv_dict.items():
            # get the coordinates of the spectrum
            self.xpos=spectrum.coordinates[0]
            self.ypos=spectrum.coordinates[1]
            # get the neighbours
            keys=self.get_neighbours()
            # get a list of neighbour spectra
            neighbours=[self.indiv_dict[key] for key in keys if key in self.indiv_dict.keys() and key is not np.nan and self.indiv_dict[key].model.ncomps != 0.0]
            # check to see if the spectrum is in our neighbour list and remove it
            if spectrum.index in [neighbour.index for neighbour in neighbours]:
                idspec=np.where([(spectrum.index == neighbour.index) for neighbour in neighbours])[0]
                del neighbours[idspec[0]]

            neighbours_ncomps=[neighbour.model.ncomps for neighbour in neighbours]
            # get weights
            weights=self.get_weights(spectrum, neighbours)

            # flags:
            setattr(spectrum,'flag',[False, False, False, False, False, False],)

            # flag zero component models
            if spectrum.model.ncomps == 0.0:
                spectrum.flag[1]=True
            else:
                if np.size(neighbours)==0:
                    spectrum.flag[2]=True
                else:
                    # flag deviations of weighted median number of components
                    wmedian_ncomps=self.weighted_median(neighbours_ncomps, weights)
                    # compute delta ncomps
                    delta_ncomps=np.abs(spectrum.model.ncomps-wmedian_ncomps)

                    # flag if delta ncomps > self.flag_deltancomps - make user defined
                    if delta_ncomps > self.flag_deltancomps:
                        spectrum.flag[3]=True

                    # flag number of component jumps
                    # determine the number of occurances where the differnce in ncomps
                    # between the spectrum and its neighbours is > self.flag_ncomponentjump
                    ncomponent_jumps = self.get_component_jumps(spectrum.model.ncomps, neighbours_ncomps)
                    # flag the pixel if the number of component jumps that are > self.flag_ncomponentjump
                    # is > self.flag_njumps
                    if ncomponent_jumps > self.flag_njumps:
                        spectrum.flag[4]=True

                    # flag spectrum properties
                    meanneighbour_params, meanneighbour_stddev = self.get_mean_neighbour(spectrum.model, [spec.model for spec in neighbours], weights)
                    nstddev,nstddev_av = self.get_nstddev_from_mean(spectrum.model, meanneighbour_params, meanneighbour_stddev)

                    if np.any(nstddev_av>self.flag_nstddev):
                        spectrum.flag[5]=True
                        # which component is flagged
                        compflag=[np.any(nstddev[j]>self.flag_nstddev) for j in range(spectrum.model.ncomps)]
                        # which parameter is flagged
                        paramflag=[(nstddev[j][k]>self.flag_nstddev) for j in range(spectrum.model.ncomps) for k in range(len(spectrum.model.parnames))]
                        setattr(spectrum,'compflag',compflag)
                        setattr(spectrum,'paramflag',paramflag)

        return self

    def flagging_indiv(self, spectrum):
        """

        """
        # get the coordinates of the spectrum
        self.xpos=spectrum.coordinates[0]
        self.ypos=spectrum.coordinates[1]
        # get the neighbours
        keys=self.get_neighbours()
        # get a list of neighbour spectra
        neighbours=[self.indiv_dict[key] for key in keys if key in self.indiv_dict.keys() and key is not np.nan and self.indiv_dict[key].model.ncomps != 0.0]

        # check to see if the spectrum is in our neighbour list and remove it
        if spectrum.index in [neighbour.index for neighbour in neighbours]:
            idspec=np.where([(spectrum.index == neighbour.index) for neighbour in neighbours])[0]
            del neighbours[idspec[0]]

        neighbours_ncomps=[neighbour.model.ncomps for neighbour in neighbours]
        # get weights
        weights=self.get_weights(spectrum, neighbours)

        # flags:
        setattr(spectrum,'flag',[False, False, False, False, False, False],)

        # flag zero component models
        if spectrum.model.ncomps == 0.0:
            spectrum.flag[1]=True
        else:
            if np.size(neighbours)==0:
                spectrum.flag[2]=True
            else:
                # flag deviations of weighted median number of components
                wmedian_ncomps=self.weighted_median(neighbours_ncomps, weights)
                # compute delta ncomps
                delta_ncomps=np.abs(spectrum.model.ncomps-wmedian_ncomps)

                # flag if delta ncomps > self.flag_deltancomps - make user defined
                if delta_ncomps > self.flag_deltancomps:
                    spectrum.flag[3]=True

                # flag number of component jumps
                # determine the number of occurances where the differnce in ncomps
                # between the spectrum and its neighbours is > self.flag_ncomponentjump
                ncomponent_jumps = self.get_component_jumps(spectrum.model.ncomps, neighbours_ncomps)
                # flag the pixel if the number of component jumps that are > self.flag_ncomponentjump
                # is > self.flag_njumps
                if ncomponent_jumps > self.flag_njumps:
                    spectrum.flag[4]=True

                # flag spectrum properties
                meanneighbour_params, meanneighbour_stddev = self.get_mean_neighbour(spectrum.model, [spec.model for spec in neighbours], weights)
                # print('mean: ', meanneighbour_params)
                nstddev,nstddev_av = self.get_nstddev_from_mean(spectrum.model, meanneighbour_params, meanneighbour_stddev)

                if np.any(nstddev_av>self.flag_nstddev):
                    spectrum.flag[5]=True
                    # which component is flagged
                    compflag=[np.any(nstddev[j]>self.flag_nstddev) for j in range(spectrum.model.ncomps)]
                    # which parameter is flagged
                    paramflag=[(nstddev[j][k]>self.flag_nstddev) for j in range(spectrum.model.ncomps) for k in range(len(spectrum.model.parnames))]
                    setattr(spectrum,'compflag',compflag)
                    setattr(spectrum,'paramflag',paramflag)

        return self

    def create_flag_map(self, save_map=True, outputfits='flagmap.fits'):
        """

        """
        self.flagmap=np.zeros(self.cubeshape[1:])*np.nan
        for key,spectrum in self.indiv_dict.items():
            cx,cy = spectrum.coordinates
            if np.any(spectrum.flag):
                index = np.where(np.asarray(spectrum.flag))[0]
                if 5 in index:
                    self.flagmap[cy, cx]=5
                else:
                    self.flagmap[cy, cx]=index[0]
            else:
                self.flagmap[cy, cx]=0

        if save_map:
            self.save_map_to_fits(outputfits)

    def refit(self):
        from .SpectralDecomposer import Decomposer
        from .model_housing import indivmodel
        from .model_housing import individual_spectrum
        from copy import deepcopy
        import time
        # create a list of flagged spectra
        # get the length of the list
        # stop == False
        # while stop == False
        # find the number of flagged spectra that have at least 1 none flagged neighbours
        # if n !=0:
        # sort the flagged spectra by the number of none flagged neighbours
        # loop
        # get the first
        # get the median number of components in the non flagged neighbours
        # get the mean model from those neighbours weighted by distance
        # refit flagged spectrum with input guesses from neighbours
        # test flag
        # if flagging satisfied:
        # update model
        # else:
        # add to new list of flagged spectra
        # at the end of the loop:
        # set the flagged spec list to the new one
        # if the length of the new list is the same as the old list:
        # stop == True

        # create a list of flagged spectra
        flag = [spectrum for key, spectrum in self.indiv_dict.items() if np.any(spectrum.flag)]
        # length of this list (used for stopping criteria)
        lenflag = np.size(flag)
        stop=False

        while stop==False:
            # sort the list of flagged spectra according to the number of non-flagged
            # neighbours
            sortedflag, sortednumneighbours = self.sortflag(flag)
            # prepare the newflag list with spectra that do not have non-flagged neighbours
            newflag = [spectrum for key, spectrum in enumerate(sortedflag) if sortednumneighbours[key]==0.0]
            # remove spectra that do not have non-flagged neighbours from the sortedflag list
            sortedflag = [spectrum for key, spectrum in enumerate(sortedflag) if sortednumneighbours[key]!=0.0]

            print('')
            print('refit: ', lenflag)
            print('sortedflag: ', len(sortedflag))
            print('newflag: ', len(newflag))

            for key, spectrum in enumerate(sortedflag):

                # get the non-flagged neighbours
                neighbours = self.get_non_flagged_neighbours(spectrum)
                # get the number of components in the non-flagged neighbours
                neighbours_ncomps=[neighbour.model.ncomps for neighbour in neighbours]
                # get weights
                weights=self.get_weights(spectrum, neighbours)
                # get the weighted median number of ncomps
                wmedian_ncomps=self.weighted_median(neighbours_ncomps, weights)
                # get the mean neighbour properties
                if spectrum.model.ncomps!=0:
                    meanneighbour_params, meanneighbour_stddev = self.get_mean_neighbour(spectrum.model, [spec.model for spec in neighbours], weights)
                    replacement_model_found, spectrum = self.check_model_bank(spectrum, wmedian_ncomps)
                else:
                    meanneighbour_params, meanneighbour_stddev = self.get_mean_neighbour_zerocomps(spectrum, neighbours, wmedian_ncomps)
                    replacement_model_found=False

                if not replacement_model_found:
                    if len(meanneighbour_params)!=0:
                        # set up the decomposer
                        decomposer=Decomposer(self.spectral_axis,spectrum.spectrum[self.trimids],spectrum.rms)
                        Decomposer.create_a_spectrum(decomposer)
                        # fit the spectrum
                        model=Decomposer.fit_spectrum_with_guesses(decomposer,meanneighbour_params,fittype='gaussian', method='spatial')
                        # generate a model
                        model=indivmodel(decomposer.modeldict)
                        # deep copy spectrum
                        spectrum_=deepcopy(spectrum)
                        setattr(spectrum_, 'model', model)
                        # check if flagging satisfied
                        self.flagging_indiv(spectrum_)
                        if np.any(spectrum_.flag):
                            self.flagging_indiv(spectrum)
                            newflag.append(spectrum)
                        else:
                            rms_passed=self.check_rms(spectrum_)
                            width_passed=self.check_width(spectrum_)
                            if rms_passed & width_passed:
                                setattr(spectrum, 'model', model)
                                individual_spectrum.add_model(spectrum, model)
                                self.flagging_indiv(spectrum)
                            else:
                                self.flagging_indiv(spectrum)
                                newflag.append(spectrum)
                    else:
                        self.flagging_indiv(spectrum)
                        newflag.append(spectrum)

            if len(newflag)==lenflag:
                stop=True
            else:
                flag = newflag
                lenflag=len(flag)
        print('')

    def sortflag(self, flag):

        numneighbours=[]
        flag = np.asarray(flag)
        for index, spectrum in enumerate(flag):
            # get the coordinates of the spectrum
            self.xpos=spectrum.coordinates[0]
            self.ypos=spectrum.coordinates[1]
            # get the neighbours
            keys=self.get_neighbours()
            # get a list of neighbour spectra that they themselves are not flagged
            nfneighbours=[self.indiv_dict[key] for key in keys if key in self.indiv_dict.keys() and key is not np.nan and np.invert(np.any(self.indiv_dict[key].flag))]
            # check to see if the spectrum is in our neighbour list and remove it
            if spectrum.index in [nfneighbour.index for nfneighbour in nfneighbours]:
                idspec=np.where([(spectrum.index == nfneighbour.index) for nfneighbour in nfneighbours])[0]
                del nfneighbours[idspec[0]]
            numneighbours.append(np.size(nfneighbours))

        numneighbours=np.asarray(numneighbours)
        sortedids = np.argsort(numneighbours)[::-1]
        sortedflag, sortednumneighbours=flag[sortedids], numneighbours[sortedids]

        return sortedflag, sortednumneighbours

    def get_non_flagged_neighbours(self, spectrum):
        # get the coordinates of the spectrum
        self.xpos=spectrum.coordinates[0]
        self.ypos=spectrum.coordinates[1]
        # get the neighbours
        keys=self.get_neighbours()

        # get a list of neighbour spectra that they themselves are not flagged
        nfneighbours=[self.indiv_dict[key] for key in keys if key in self.indiv_dict.keys() and key is not np.nan and np.invert(np.any(self.indiv_dict[key].flag))]
        # check to see if the spectrum is in our neighbour list and remove it
        if spectrum.index in [nfneighbour.index for nfneighbour in nfneighbours]:
            idspec=np.where([(spectrum.index == nfneighbour.index) for nfneighbour in nfneighbours])[0]
            del nfneighbours[idspec[0]]

        return nfneighbours

    def check_model_bank(self, spectrum, wmedian_ncomps):
        from copy import deepcopy
        replacement_model_found=False
        flags=[]
        ncomps=[]

        # print('----')
        # print(spectrum)
        # print('----')

        # first check to see if the current model is in the model bank and
        # remove
        alternative_models=deepcopy(spectrum.model_from_parent)
        if spectrum.model in alternative_models:
            alternative_models.remove(spectrum.model)

        if None in alternative_models:
            alternative_models.remove(None)

        if np.size(alternative_models)==0:
            flags=[False]
        else:
            # first loop over the models to see if any of them satisfy the flag
            # criertia
            for i, model in enumerate(alternative_models):
                # get the number of components
                ncomps.append(model.ncomps)
                # create a copy of the spectrum
                spectrum_=deepcopy(spectrum)
                # update the selected model with the new model
                setattr(spectrum_, 'model', model)
                # run flagging
                # print('flagging')
                self.flagging_indiv(spectrum_)
                # print(spectrum_.flag)
                # check to see if the criteria is satisfied
                if np.any(spectrum_.flag):
                    flags.append(False)
                else:
                    flags.append(True)

            # if there are any cases where a suitable replacement has been found
            # we are going to update the model

            # print(flags)
            # print(alternative_models)
            if np.any(flags):
                replacement_model_found=True
                # create a copy of the spectrum
                spectrum_=deepcopy(spectrum)
                # check to see if any of the suitable models have the same number of
                # components as the median ncomps from the neighbours
                id = np.where((np.asarray(ncomps)==wmedian_ncomps) & (flags==True))[0]
                if np.size(id)!=0:
                    # if that is true add the first one that satisfies the condition
                    # (there could be more than one)

                    # update the model list with the old model
                    if spectrum.model not in spectrum.model_from_parent:
                        spectrum.model_from_parent.append(spectrum.model)
                    # update the favoured model from the model list of the copied
                    # spectrum
                    spectrum.model=alternative_models[id[0]]
                else:
                    # if that is not true then we are going to still swap the model.
                    # loop over the possible models
                    for i, model in enumerate(alternative_models):
                        if flags[i]:
                            if spectrum.model not in spectrum.model_from_parent:
                                spectrum.model_from_parent.append(spectrum.model)
                            spectrum.model=alternative_models[i]

            # re run flagging
            self.flagging_indiv(spectrum)
            # print(spectrum.flag)
            # print('')

        return replacement_model_found, spectrum

    def save_map_to_fits(self, outputfits):
        """

        """
        from astropy.io import fits
        savedir=self.outputdirectory+self.filename+'/stage_4/'

        fh = fits.PrimaryHDU(data=self.flagmap, header=self.header)
        fh.writeto(os.path.join(savedir, outputfits), overwrite=True)

    def save_flags(self, outputfile='s4.flags.scousepy'):
        """

        """
        import pickle
        if os.path.exists(self.outputdirectory+self.filename+'/stage_4/'+outputfile):
            os.rename(self.outputdirectory+self.filename+'/stage_4/'+outputfile,self.outputdirectory+self.filename+'/stage_4/'+outputfile+'.bk')

        with open(self.outputdirectory+self.filename+'/stage_4/'+outputfile, 'wb') as fh:
            pickle.dump((self), fh, protocol=proto)

    @staticmethod
    def load_flags(fn):
        """

        """
        import pickle
        with open(fn, 'rb') as fh:
            self = pickle.load(fh)

        return self

    def stats(self):
        """

        """
        statkeys=['flaggedpixels', 'nflags','zerocomp','noneighbours','deltancomps','ncompjumps','sigmadiff']

        flaggedpixels=np.count_nonzero([np.any(spectrum.flag) for key, spectrum in self.indiv_dict.items()])
        nflags=np.sum([np.size(np.where(np.asarray(spectrum.flag))[0]) for key, spectrum in self.indiv_dict.items()])
        zerocomp=np.count_nonzero([(spectrum.flag[1]) for key, spectrum in self.indiv_dict.items()])
        noneighbours=np.count_nonzero([(spectrum.flag[2]) for key, spectrum in self.indiv_dict.items()])
        deltancomps=np.count_nonzero([(spectrum.flag[3]) for key, spectrum in self.indiv_dict.items()])
        ncompjumps=np.count_nonzero([(spectrum.flag[4]) for key, spectrum in self.indiv_dict.items()])
        sigmadiff=np.count_nonzero([(spectrum.flag[5]) for key, spectrum in self.indiv_dict.items()])

        flags=[flaggedpixels,nflags,zerocomp,noneighbours,deltancomps,ncompjumps,sigmadiff]
        statdict={}
        for i, key in enumerate(statkeys):
            statdict[key]=flags[i]

        return statdict

    def get_neighbours(self):
        """
        Returns a list of flattened indices for a given spectrum and its neighbours

        """
        shape=self.cubeshape[1:]
        neighboursx=np.arange(self.xpos-(self.blocksize-1)/2,(self.xpos+(self.blocksize-1)/2)+1,dtype='int' )
        neighboursx=[x if (x>=0) & (x<=shape[1]-1) else np.nan for x in neighboursx ]
        neighboursy=np.arange(self.ypos-(self.blocksize-1)/2,(self.ypos+(self.blocksize-1)/2)+1,dtype='int' )
        neighboursy=[y if (y>=0) & (y<=shape[0]-1) else np.nan for y in neighboursy ]
        keys=[np.ravel_multi_index([y,x], shape)  if np.all(np.isfinite(np.asarray([y,x]))) else np.nan for y in neighboursy for x in neighboursx]

        return keys

    def get_weights(self, spectrum, neighbours):
        """

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
        https://gist.github.com/tinybike/d9ff1dad515b66cc0d87
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

    def check_rms(self, spectrum):
        """
        Check the rms of the best-fitting model components

        Parameters
        ----------
        condition_passed : list
            boolean list indicating which quality control steps have been satisfied

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

        # Now check all components to see if they are above the rms threshold
        for i in range(int(ncomponents)):
            if (spectrum.model.params[int(i*nparams)+idx] < spectrum.rms*self.tol[1]):
                return False

        return True

    def check_width(self, spectrum):
        """
        Check the fwhm of the best-fitting model components

        Parameters
        ----------
        condition_passed : list
            boolean list indicating which quality control steps have been satisfied

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

        # Now check all components to see if they are above the rms threshold
        for i in range(int(ncomponents)):
            if (spectrum.model.params[int((i*nparams)+idx)]*fwhmconv < self.res*self.tol[2]):
                return False

        return True
