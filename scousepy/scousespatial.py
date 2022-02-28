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

    def flagging(self, scouseobject):
        """

        """
        counter=0.0
        for index, spectrum in self.indiv_dict.items():
            # get the coordinates of the spectrum
            self.xpos=spectrum.coordinates[0]
            self.ypos=spectrum.coordinates[1]
            # get the neighbours
            keys=self.get_neighbours()
            # get a list of neighbour spectra
            neighbours=[self.indiv_dict[key] for key in keys if key in self.indiv_dict.keys() and key is not np.nan and self.indiv_dict[key].model.ncomps != 0.0]
            # check to see if the spectrum is in our neighbour list and remove it
            if spectrum in neighbours:
                neighbours.remove(spectrum)
            neighbours_ncomps=[neighbour.model.ncomps for neighbour in neighbours]
            # get weights
            weights=self.get_weights(spectrum, neighbours)

            # flags:
            setattr(spectrum,'flag',[False, False, False, False, False, False],)

            # flag zero component models
            if spectrum.model.ncomps == 0.0:
                spectrum.flag[1]=True
                # print('')
                # print("index: ", spectrum.index)
                # print("coordinates: ", spectrum.coordinates)
                # print("spectrum: ", spectrum)
                # print("rms: ", spectrum.rms)
                # print("model: ", spectrum.model)
                # print('')
                # print("guesses from parent: ", spectrum.guesses_from_parent)
                # print("guesses updated: ", spectrum.guesses_updated)
                # print('')
                #
                # from .SpectralDecomposer import Decomposer
                # from .model_housing import indivmodel
                # decomposer=Decomposer(scouseobject.xtrim,spectrum.spectrum,spectrum.rms)
                # setattr(decomposer,'psktemplate',spectrum.template,)
                # if np.size(spectrum.guesses_updated)<=1:
                #     guesses=spectrum.guesses_from_parent
                # else:
                #     guesses=spectrum.guesses_updated
                #
                # guesses_parent=spectrum.guesses_from_parent
                # print("guesses from parent: ", guesses_parent)
                # print("guesses: ", guesses)
                #
                # Decomposer.fit_spectrum_from_parent(decomposer,guesses_parent,guesses_parent,[4.0, 3.0, 0.5, 5.0, 5.0, 0.5],2.54,fittype='gaussian',)
                # print(decomposer.guesses_updated)
                # print(decomposer.validfit)
                # print(decomposer.conditions)
                # print('')
                #
                # if np.size(decomposer.guesses_updated)<=1:
                #     guesses=spectrum.guesses_from_parent
                # else:
                #     guesses=decomposer.guesses_updated
                #
                # Decomposer.fit_spectrum_from_parent(decomposer,guesses,guesses_parent,[4.0, 3.0, 0.5, 5.0, 5.0, 0.5],2.54,fittype='gaussian',)
                # print(decomposer.guesses_updated)
                # print(decomposer.validfit)
                # print(decomposer.conditions)
                # import matplotlib.pyplot as plt
                # fig = plt.figure()
                # plt.plot(scouseobject.xtrim, spectrum.spectrum, drawstyle='steps', color='k', lw=0.85)
                # plt.show()
                #
                # counter+=1
                # if counter==40:
                #     sys.exit()
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
                    mean_neighbour_components, stddev_neighbour_components = self.get_mean_neighbour_component(spectrum.model, [spec.model for spec in neighbours], weights)
                    nstddev = self.get_nstddev_from_mean(spectrum.model, mean_neighbour_components, stddev_neighbour_components)
                    if np.any(nstddev>self.flag_nstddev):
                        spectrum.flag[5]=True
                        compflag=[np.any(nstddev[j]>self.flag_nstddev) for j in range(spectrum.model.ncomps)]
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

        return weighted_median

    def get_component_jumps(self, ncomps_spec, ncomps_neighbours):
        """

        """
        ncomp_jumps=np.asarray([np.abs(ncomps_spec-ncomps_neighbour) for ncomps_neighbour in ncomps_neighbours])
        return np.count_nonzero(ncomp_jumps>self.flag_ncomponentjump)

    def get_nstddev_from_mean(self,model, mean, stddev):
        """

        """
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

    def get_mean_neighbour_component(self, model, neighbour_models, weights):
        """

        """
        neighbour_components=[]
        neighbour_weights=[]
        mean_neighbour_components=[]
        stddev_neighbour_components=[]
        for _ in range(model.ncomps):
            neighbour_components.append([])
            neighbour_weights.append([])
            mean_neighbour_components.append([])
            stddev_neighbour_components.append([])

        for j in range(len(neighbour_models)):
            neighbour_components, neighbour_weights = self.find_closest_match(model, neighbour_models[j], neighbour_components, neighbour_weights, weights[j])

        for k in range(model.ncomps):
            if np.size(neighbour_components[k])!=0.0:
                for m in range(np.shape(neighbour_components[k])[1]):
                    property=np.asarray([neighbour_components[k][l][m] for l in range(np.shape(neighbour_components[k])[0])])
                    weighting=np.asarray(neighbour_weights[k])
                    idfinite=np.where(np.isfinite(property))
                    property=property[idfinite]
                    weighting=weighting[idfinite]
                    weighted_average=np.average(property, weights=neighbour_weights[k])
                    variance = np.average((np.asarray(property)-weighted_average)**2, weights=neighbour_weights[k])
                    mean_neighbour_components[k].append(weighted_average)
                    stddev_neighbour_components[k].append(np.sqrt(variance))
            else:
                for m in range(len(model.parnames)):
                    mean_neighbour_components[k].append(0.0)
                    stddev_neighbour_components[k].append(1e-10)

        return mean_neighbour_components, stddev_neighbour_components

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
            neighbour_weights[idmin].append(weight)

        return neighbour_components, neighbour_weights
