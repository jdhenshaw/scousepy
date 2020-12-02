# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2019 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np
import warnings
from astropy import log
import sys
from .colors import *
# import the decomposer
from scousepy.SpectralDecomposer import Decomposer

class ScouseFitter(object):
    """
    Interactive fitting window for scouse
    """

    def __init__(self, modelstore,
                       method='scouse',
                       spectra=None,
                       scouseobject=None,
                       fit_dict=None,
                       parent=None,
                       scouse_stage_6=False,
                       individual=None,
                       cube=None,
                       fittype='gaussian',
                       fitcount=None,
                       x=None,y=None,rms=None,
                       SNR=3,minSNR=1,maxSNR=30,
                       kernelsize=3,minkernel=1,maxkernel=30,
                       outputfile=None,
                       xarrkwargs={},unit=''):

        """

        Parameters
        ----------

        modelstore : dictionary
            A dictionary within which the model solutions will be stored.

        method : string
            Options are scouse, cube, or single. Scousefitter is generalised so
            that it can be used to fit individual spectra as well. Use
            'individual' to fit either a single spectrum or an array of spectra.
            Use 'cube' to fit a data cube.

        spectra : ndarray
            An array of indices relating to the spectra that are to be fitted.

        scouseobject : scouse class object
            For scouse framework only. Instance of the scouse object.

        fit_dict : dictionary
            For scouse framework only. Dictionary of SAAs. Used to
            retrieve spectra.

        parent : ndarray
            For scouse framework only. Array referencing the parent SAA. Used to
            retrieve spectra.

        scouse_single : bool
            True or False. Used during stage 6 of scouse where individual
            spectra are fitted rather that SAAs.

        individual : ndarray
            An array of spectra to be fit. Should be in the format [n,2,m].
            Where n is the number of spectra, 2 corresponds to the
            velocity/frequency and intensity axes, each of which have length m.
            This must be provided with method='individual'.

        cube : ndarray
            A cube of data to be fit. Should be a 3D array. This must be
            provided if method='cube'.

        fitcount : boolean array
            Used to keep tabs on the fitting. An array of equivalent length to
            the number of spectra. Updated as each spectrum is fitted.

        SNR : number
            Initial signal-to-noise ratio (SNR) used for derivative spectroscopy
            automated fitting.

        minSNR : number
            Minimum SNR. Used for plotting. Can be adjusted and will change the
            slider values.

        maxSNR : number
            Maximum SNR. Used for plotting. Can be adjusted and will change the
            slider values.

        kernelsize : number
            Initial kernel size used for derivative spectroscopy automated
            fitting. Provided in integer number of channels over which to smooth.

        minkernel : number
            Minimum kernel size. Used for plotting. Can be adjusted and will
            change the slider values.

        maxkernel : number
            Maximum kernel size. Used for plotting. Can be adjusted and will
            change the slider values.

        """

        # set the global quantities
        self.method=method
        self.spectra=spectra
        self.scouseobject=scouseobject
        self.fit_dict=fit_dict
        self.parent=parent
        self.scouse_stage_6=scouse_stage_6
        self.individual=individual
        self.cube=cube
        self.fittype=fittype
        self.fitcount=fitcount
        self.decomposer=None
        self.modelstore=modelstore
        self.SNR=SNR
        self.minSNR=minSNR
        self.maxSNR=maxSNR
        self.kernelsize=kernelsize
        self.minkernel=minkernel
        self.maxkernel=maxkernel
        self.outputfile=outputfile
        self.models={}

        # Prepare the fitter according to method selection
        if method=='scouse':
            # set some defaults for pyspeckit
            import astropy.units as u
            self.xarrkwargs={'unit':'km/s',
                             'refX': self.scouseobject.cube.wcs.wcs.restfrq*u.Hz,
                             'velocity_convention': 'radio'}
            self.unit=self.scouseobject.cube.header['BUNIT']

            # For scouse fitting
            if (scouseobject is None) or (spectra is None):
                # make sure that the scouse object has been sent
                ValueError(colors.fg._red_+"Please include both the scousepy object and the spectra to be fit."+colors._endc_)
            else:
                # Create an array of indices for the spectra to be fitted.
                self.indexlist=np.arange(0,int(np.size(self.spectra)))

                # index of the first spectrum to be fit. First establish if
                # any of the spectra have been fitted aleady.
                if np.any(self.fitcount):
                    # Check to see if all of the spectra have been fitted.
                    if np.all(self.fitcount):
                        print('')
                        print(colors.fg._lightgreen_+"All spectra have solutions. Fitting complete. "+colors._endc_)
                        print('')
                        # exit if fitting has been completed
                        return
                    else:
                        # pick up from where you left off
                        self.index=np.where(self.fitcount==False)[0][0]
                else:
                    # start at the beginning
                    self.index=0

                # retrieve the scouse spectrum from the scouse dictionary
                self.my_spectrum=retrieve_spectrum(self,self.spectra,self.index)
                # get the x,y,rms values
                get_spectral_info(self)
                # initiate the decomposer
                self.initiate_decomposer()

        # Cube fitting
        elif method=='cube':

            # TODO: Implement cube fitting

            # pyspeckit kwargs
            self.xarrkwargs=xarrkwargs
            self.unit=unit
            # assert that the user has provided the individual spectra
            assert cube is not None, colors.fg._red_+"Please include the data cube to be fit."+colors._endc_


        # fitting individual spectra
        elif method=='individual':
            # pyspeckit kwargs
            self.xarrkwargs=xarrkwargs
            self.unit=unit
            # assert that the user has provided the individual spectra
            assert individual is not None, colors.fg._red_+"Please include an array of spectra to be fit."+colors._endc_
            # get the shape of the input array
            shape=np.shape(individual)
            # create a boolean array to be updated as the spectra are fitted
            self.fitcount=np.zeros(shape[0], dtype='bool')
            # Create an array of indices for the spectra to be fitted.
            self.indexlist=np.arange(0,int(shape[0]))
            self.spectra=self.indexlist
            # start at the beginning
            self.index=0
            # get the x,y,rms values
            get_spectral_info(self)
            # initiate the decomposer
            self.initiate_decomposer()

        else:
            # throw an error
            ValueError("Please use a valid method type: 'scouse', 'cube', 'individual'")

        # imports
        from scousepy.dspec import DSpec
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        self.cmap=plt.cm.binary_r
        rcParams['font.family']= 'Arial'
        rcParams['font.size']= 9
        rcParams['lines.linewidth']= 1.     ## line width in points
        rcParams['axes.labelsize'] =10  ## fontsize of the x any y labels
        rcParams['xtick.labelsize']=10 ## fontsize of the tick labels
        rcParams['ytick.labelsize'] =10 ## fontsize of the tick labels
        rcParams['xtick.major.pad']=8   ## distance to major tick label in points
        rcParams['ytick.major.pad']=8    ## distance to major tick label in points
        rcParams['xtick.major.size'] =4    ## major tick size in points
        rcParams['xtick.minor.size' ]=2     ## minor tick size in points
        rcParams['xtick.major.width'] =1.    ## major tick width in points
        rcParams['xtick.minor.width']=1.    ## minor tick width in points
        rcParams['ytick.major.size']= 4    ## major tick size in points
        rcParams['ytick.minor.size' ]=2      ## minor tick size in points
        rcParams['ytick.major.width']=1.    ## major tick width in points
        rcParams['ytick.minor.width']= 1.    ## minor tick width in points

        # remove some matplotlib keyboard shortcuts to prevent meltdown
        plt.rcParams['keymap.quit'].remove('q')
        plt.rcParams['keymap.quit_all'].remove('Q')

        # compute derivative spectroscopy for spectrum in memory
        self.dsp = compute_dsp(self)

        #================#
        # initiate the GUI
        #================#
        self.fig = plt.figure(figsize=(14, 8))

        #===============#
        # spectrum window
        #===============#
        # Set up the plot defaults
        self.spectrum_window_ax=[0.05,0.475,0.375,0.425]
        ymin=np.min(self.specy)-0.2*np.max(self.specy)
        ymax=np.max(self.specy)+0.2*np.max(self.specy)
        self.spectrum_window=setup_plot_window(self,self.spectrum_window_ax,ymin=ymin,ymax=ymax)
        self.spectrum_window.text(0.99, 0.05, 'select legend items to toggle',
                                  transform=self.spectrum_window.transAxes,
                                  fontsize=8, ha='right')
        # plot the spectrum and the smoothed spectrum
        self.plot_spectrum,=plot_spectrum(self,self.specx,self.specy,label='spec')
        self.plot_smooth,=plot_spectrum(self,self.specx,self.ysmooth,label='smoothed spec',lw=1.5,ls=':',color='k')
        # plot the signal to noise threshold
        self.plot_SNR,=plot_snr(self,linestyle='--',color='k',label='SNR*rms')
        # plot the predicted peak locations
        self.plot_peak_markers,=plot_peak_locations(self,color='k',linestyle='',marker='o',markersize=5)
        # plot stems for the markers
        self.plot_peak_lines=plot_stems(self,color='k')

        #============#
        # deriv window
        #============#
        # Set up the plot defaults
        self.deriv_window_ax=[0.45,0.475,0.375,0.425]
        ymin=np.min([np.min(self.d1/np.max(self.d1)),np.min(self.d2/np.max(self.d2)),np.min(self.d3/np.max(self.d3)),np.min(self.d4/np.max(self.d4))])
        ymax=np.max([np.max(self.d1/np.max(self.d1)),np.max(self.d2/np.max(self.d2)),np.max(self.d3/np.max(self.d3)),np.max(self.d4/np.max(self.d4))])
        lim=np.max(np.abs([ymin,ymax]))
        self.deriv_window=setup_plot_window(self,self.deriv_window_ax,ymin=-1*lim,ymax=lim)
        # plot the (normalised) derivatives
        plot_derivatives(self)
        # plot a legend
        self.deriv_legend = self.deriv_window.legend(loc=2,frameon=False,fontsize=8)
        self.deriv_window_lookup_artist, self.deriv_window_lookup_handle = self.build_legend_lookups(self.deriv_legend)
        self.setup_legend_connections(self.deriv_legend, self.deriv_window_lookup_artist, self.deriv_window_lookup_handle)
        self.update_legend(self.deriv_window_lookup_artist, self.deriv_window_lookup_handle)

        #==================#
        # information window
        #==================#
        self.information_window_ax=[0.05, 0.08, 0.9, 0.35]
        setup_information_window(self)
        self.text_snr=print_information(self,0.01,0.84,'SNR: '+str(self.SNR), fontsize=10)
        self.text_kernel=print_information(self,0.01,0.76,'kernel size: '+str(self.kernelsize),fontsize=10)
        self.text_fitinformation=print_information(self,0.01,0.68,'pyspeckit fit information: ', fontsize=10, fontweight='bold')
        self.text_converge=print_information(self, 0.15,0.68, '', fontsize=10, color='green', fontweight='bold')
        self.text_ncomp=print_information(self,0.01,0.6,'number of components: '+str(self.ncomps),fontsize=10)
        self.text_peaks=print_information(self,0.01,0.52,'', fontsize=10)
        self.text_centroids=print_information(self,0.01,0.44,'', fontsize=10)
        self.text_widths=print_information(self,0.01,0.36,'', fontsize=10)
        self.text_goodnessformation=print_information(self,0.01,0.28,'goodness of fit information: ', fontsize=10, fontweight='bold')
        self.text_chisq=print_information(self,0.01,0.2,'', fontsize=10)
        self.text_redchisq=print_information(self,0.01,0.12,'', fontsize=10)
        self.text_aic=print_information(self,0.01,0.04,'', fontsize=10)

        #=======================#
        # Fitting and information
        #=======================#
        self.residkwargs={'color':'orange','ls':'-','lw':1}
        self.modelkwargs={'color':'limegreen','ls':'-','lw':1}
        self.totmodkwargs={'color':'magenta','ls':'-','lw':1}

        # Fit the spectrum according to dspec guesses
        if not self.scouse_stage_6 and self.dsp.ncomps != 0:
            Decomposer.fit_spectrum_with_guesses(self.decomposer,self.guesses,fittype=self.fittype)

        # if stage 6 is being run we don't want to fit we want to populate a
        # dictionary of pre-existing models
        if self.scouse_stage_6:
            populate_models(self)

        # Add the best-fitting solution and useful parameters to a dictionary
        self.modeldict=get_model_info(self)
        # Recreate the model
        self.mod,self.res,self.totmod=recreate_model(self)
        # Update the plot adding the model
        update_plot_model(self)
        # Print the fit information to the information window
        print_fit_information(self)

        #=======#
        # buttons
        #=======#
        self.button_previous_ax=self.fig.add_axes([0.85, 0.85, 0.1, 0.05])
        self.button_previous=make_button(self.button_previous_ax,"previous",lambda event: self.new_spectrum(event, _type='previous'))

        self.button_next_ax=self.fig.add_axes([0.85, 0.775, 0.1, 0.05])
        self.button_next=make_button(self.button_next_ax,"next",lambda event: self.new_spectrum(event, _type='next'))

        self.button_fit_dspec_ax=self.fig.add_axes([0.85, 0.7, 0.1, 0.05])
        self.button_fit_dspec=make_button(self.button_fit_dspec_ax,"fit (dspec)",self.dspec_manual)

        self.button_fit_manual_ax=self.fig.add_axes([0.85, 0.625, 0.1, 0.05])
        self.button_fit_manual=make_button(self.button_fit_manual_ax,"fit (manual)",self.open_scousefitter_manual)

        self.button_applyall_ax=self.fig.add_axes([0.85, 0.55, 0.1, 0.05])
        self.button_applyall=make_button(self.button_applyall_ax,"apply dspec to all",self.dspec_apply_to_all)

        self.button_stop_ax=self.fig.add_axes([0.85, 0.475, 0.1, 0.05])
        self.button_stop=make_button(self.button_stop_ax,"exit",self.stop, color='lightblue',hovercolor='aliceblue')

        # navigation buttons
        inc=0.08

        self.button_skiptostart=self.fig.add_axes([0.3+inc, 0.03, 0.025, 0.025])
        self.button_skiptostart=make_button(self.button_skiptostart,"<<",lambda event: self.new_spectrum(event, _type='start'))

        self.button_skipbackone=self.fig.add_axes([0.3275+inc, 0.03, 0.025, 0.025])
        self.button_skipbackone=make_button(self.button_skipbackone,"<",lambda event: self.new_spectrum(event, _type='previous'))

        self.textbox_index=self.fig.add_axes([0.365+inc, 0.03, 0.025, 0.025])
        self.textbox_index=make_textbox(self.textbox_index,str(self.index+1),self.change_text)
        self.text_totalspectra=self.fig.text(0.3925+inc,0.03725,'of '+str(len(self.spectra)))

        self.button_skipforwardone=self.fig.add_axes([0.43+inc, 0.03, 0.025, 0.025])
        self.button_skipforwardone=make_button(self.button_skipforwardone,">",lambda event: self.new_spectrum(event, _type='next'))

        self.button_skiptoend=self.fig.add_axes([0.4575+inc, 0.03, 0.025, 0.025])
        self.button_skiptoend=make_button(self.button_skiptoend,">>",lambda event: self.new_spectrum(event, _type='end'))

        #=======#
        # sliders
        #=======#
        self.slider_snr_ax=self.fig.add_axes([0.0875, 0.91, 0.3, 0.02])
        self.slider_snr=make_slider(self.slider_snr_ax,"SNR",self.minSNR,self.maxSNR,self.update_SNR,valinit=self.SNR, valfmt="%i")

        self.slider_kernel_ax=self.fig.add_axes([0.4875, 0.91, 0.3, 0.02])
        self.slider_kernel=make_slider(self.slider_kernel_ax,"kernel",self.minkernel,self.maxkernel,self.update_kernelsize,valinit=self.kernelsize, valfmt="%i")

    def show(self):
        """
        Show the plot
        """
        import matplotlib.pyplot as plt
        plt.show()

    def initiate_decomposer(self):
        """
        initiates an instance of the SpectralDecomposer
        """
        # create the decomposer
        self.decomposer=Decomposer(self.specx, self.specy, self.specrms)
        Decomposer.create_a_spectrum(self.decomposer,unit=self.unit,xarrkwargs=self.xarrkwargs)
        # generate pyspeckit spectrum
        self.spectrum=self.decomposer.pskspectrum

    def open_scousefitter_manual(self, event):
        """
        This controls the manual fitting if no reasonable solution can be found
        with the derivative spectroscopy method

        """
        # manual fit
        Decomposer.fit_spectrum_manually(self.decomposer, fittype=self.fittype)
        # add model to dictionary
        self.modeldict=get_model_info(self)
        # recreate the model
        self.mod,self.res,self.totmod=recreate_model(self)
        # update the plot with the manually-fitted solution
        update_plot_model(self,update=True)
        # update the information window
        update_text(self.text_ncomp, 'number of components: '+str(self.modeldict['ncomps']))
        print_fit_information(self)
        # update plot
        self.fig.canvas.draw()

    def dspec_manual(self, event):
        """
        This controls the manual dspec fitter.

        """
        #compute new dsp
        self.dsp = compute_dsp(self)
        # update spectrum plot
        self.plot_peak_markers=plot_peak_locations(self,update=True,plottoupdate=self.plot_peak_markers)
        self.plot_peak_lines=plot_stems(self,update=True,color='k')
        # Fit the spectrum according to dspec guesses
        if self.dsp.ncomps != 0:
            Decomposer.fit_spectrum_with_guesses(self.decomposer,self.guesses,fittype=self.fittype)
        # Add the best-fitting solution and useful parameters to a dictionary
        self.modeldict=get_model_info(self)
        # recreate the model
        self.mod,self.res,self.totmod=recreate_model(self)
        # update the plot with the dspec-fitted solution
        update_plot_model(self,update=True)
        # update the information window
        update_text(self.text_ncomp, 'number of components: '+str(self.modeldict['ncomps']))
        print_fit_information(self)

        # update plot
        self.fig.canvas.draw()

    def dspec_apply_to_all(self,event):
        """
        controls what happens if the apply dspec to all button is pressed - will
        save the current slider values apply these to all spectra which currently
        do not have solutions.

        """
        from tqdm import tqdm
        # this feature will apply dspec settings to all spectra without
        # solutions and exit the fitter - first we want to make sure we save
        # the current solution.
        print("Fitting all remaining spectra using derivative spectroscopy... ")
        print('')

        self.modelstore[self.index]=self.modeldict
        self.fitcount[self.index]=True
        # identify all spectra that do not currently have best-fitting solutions
        id = np.where(self.fitcount==False)[0]
        # loop through and fit
        for i in tqdm(id):
            # retrieve the new index
            index=int(i)
            # check against modelstore to see if there is a solution
            if not index in self.modelstore.keys():
                if self.method=='scouse':
                    # get the relevant spectrum
                    self.my_spectrum=retrieve_spectrum(self,self.spectra,index)
                else:
                    self.index=index
                # get the spectral information
                get_spectral_info(self)
                # initiate the decomposer
                self.initiate_decomposer()
                #compute new dspec
                self.dsp = compute_dsp(self)
                # Fit the spectrum according to dspec guesses
                if self.dsp.ncomps != 0:
                    Decomposer.fit_spectrum_with_guesses(self.decomposer,self.guesses,fittype=self.fittype)
                # Add the best-fitting solution and useful parameters to a dictionary
                self.modeldict=get_model_info(self)
                # add model to model store
                self.modelstore[index]=self.modeldict
                # update fit status
                self.fitcount[index]=True

        # return model dict to none
        self.modeldict=None
        # close the fitter
        self.close_window()
        # print completion statement
        print('')
        print(colors.fg._lightgreen_+"All spectra have solutions. Fitting complete. "+colors._endc_)

        return

    def stop(self, event):
        """
        Controls what happens if stop button is pressed. Can stop/start fitter
        whenever - replaces bitesize fitting in scousepy v1.
        """
        # always save the current solution before stopping the fitter
        self.modelstore[self.index]=self.modeldict
        self.fitcount[self.index]=True
        # print completion statement. If fitting has not completed throw a
        # warning to let the user know.
        if np.all(self.fitcount):
            print('')
            print(colors.fg._green_+"All spectra have solutions. Fitting complete. "+colors._endc_)

        else:
            print('')
            print(colors.fg._yellow_+"Warning: Fitting stopped. Not all spectra have solutions.  "+colors._endc_)

        # close the fitter
        self.close_window()
        # output fits for individual spectra
        if self.method=='individual':
            if self.outputfile is not None:
                print_to_file(self)
        return

    def close_window(self):
        """
        Closes the plot window
        """
        import matplotlib.pyplot as plt
        plt.rcParams['keymap.quit'].append('q')
        plt.rcParams['keymap.quit_all'].append('Q')
        plt.close('all')

    def change_text(self, text):
        """
        Controls the navigation through the fitter. Navigation controlled by the
        buttons at the bottom of the fitter. User has the option to skip forward,
        backward, to the end or back to the start. User can also input individual
        indices.

        """
        # extract value from text input.
        value = eval(text)
        # correct the indexing from base 1 to base 0 (with the former making
        # more sense for the navigator)
        value = value-1
        # Control what happens if you are at the start or end
        if value < 0:
            value=0
        if value > len(self.indexlist)-1:
            value=len(self.indexlist)-1
        # always save the current model if we move on
        if self.modeldict is not None:
            self.modelstore[self.index]=self.modeldict
            # update fit status
            self.fitcount[self.index]=True
        else:
            # if modeldict is empty then create it
            self.modeldict=get_model_info(self)
        # set the index to the new one selected by the navigator
        self.index=value
        # get the relevant spectrum
        self.my_spectrum=retrieve_spectrum(self,self.spectra,self.index)
        # get the spectral information
        get_spectral_info(self)

        # initiate the decomposer
        self.initiate_decomposer()
        #compute new dspec
        self.dsp = compute_dsp(self)

        # update spectrum plot
        ymax=np.max([self.SNR*self.specrms, np.max(self.specy)])+\
             0.2*np.max([self.SNR*self.specrms, np.max(self.specy)])
        ymin=np.min(self.specy)-0.2*np.max(self.specy)
        self.spectrum_window.set_ylim([ymin,ymax])
        self.plot_spectrum=plot_spectrum(self,self.specx,self.specy,update=True,plottoupdate=self.plot_spectrum)
        self.plot_smooth=plot_spectrum(self,self.specx,self.ysmooth,update=True,plottoupdate=self.plot_smooth)
        self.plot_SNR.set_xdata([np.min(self.specx),np.max(self.specx)])
        self.plot_SNR.set_ydata([self.SNR*self.specrms,self.SNR*self.specrms])
        self.plot_peak_markers=plot_peak_locations(self,update=True,plottoupdate=self.plot_peak_markers)
        self.plot_peak_lines=plot_stems(self,update=True,color='k')
        # update deriv plot
        ymin=np.min([np.min(self.d1/np.max(self.d1)),np.min(self.d2/np.max(self.d2)),np.min(self.d3/np.max(self.d3)),np.min(self.d4/np.max(self.d4))])
        ymax=np.max([np.max(self.d1/np.max(self.d1)),np.max(self.d2/np.max(self.d2)),np.max(self.d3/np.max(self.d3)),np.max(self.d4/np.max(self.d4))])
        lim=np.max(np.abs([ymin,ymax]))
        plot_derivatives(self,update=True,ymin=-1*lim,ymax=lim)
        # update information window
        update_text(self.text_snr, 'SNR: '+str(self.SNR))

        # check to see if a model already exists
        if self.index in self.modelstore.keys():
            # Fit the spectrum according to dspec guesses
            if self.dsp.ncomps != 0:
                Decomposer.fit_spectrum_with_guesses(self.decomposer,self.guesses,fittype=self.fittype)
            # retrieve the current model
            self.modeldict=self.modelstore[self.index]
            # recreate the model
            self.mod,self.res,self.totmod=recreate_model(self)
            # display the model
            update_plot_model(self,update=True)
            # update the information window
            update_text(self.text_ncomp, 'number of components: '+str(self.modeldict['ncomps']))
            print_fit_information(self)
        else:
            # Fit the spectrum according to dspec unless dspec cannot find any
            # components - If this happens a zero component model will be
            # displayed
            if self.dsp.ncomps!=0:
                Decomposer.fit_spectrum_with_guesses(self.decomposer,self.guesses,fittype=self.fittype)
            # Get the model
            self.modeldict=get_model_info(self)
            # recreate the model
            self.mod,self.res,self.totmod=recreate_model(self)
            # display the model
            update_plot_model(self,update=True)
            # update the information window
            update_text(self.text_ncomp, 'number of components: '+str(self.modeldict['ncomps']))
            print_fit_information(self)

        # update plot
        self.fig.canvas.draw()

    def new_spectrum(self,event,_type=None):
        """
        This descibes what happens when a new spectrum is selected using the
        buttons "previous" and "next"

        Parameters
        ----------
        event : button click event
        _type : describes what button has been pressed

        """
        value=update_index(self,_type)
        value += 1
        self.textbox_index.set_val(str(value))

    def update_SNR(self,pos=None):
        """
        This controls what happens if the SNR slider is updated

        Parameters:
        -----------
        pos : position on the slider

        """
        # New SNR
        self.SNR = int(round(pos))
        #compute new dsp
        self.dsp = compute_dsp(self)
        # update spectrum plot
        ymax=np.max([self.SNR*self.specrms, np.max(self.specy)])+\
             0.2*np.max([self.SNR*self.specrms, np.max(self.specy)])
        ymin=np.min(self.specy)-0.2*np.max(self.specy)
        self.spectrum_window.set_ylim([ymin,ymax])
        self.plot_SNR.set_xdata([np.min(self.specx),np.max(self.specx)])
        self.plot_SNR.set_ydata([self.SNR*self.specrms,self.SNR*self.specrms])
        self.plot_peak_markers=plot_peak_locations(self,update=True,plottoupdate=self.plot_peak_markers)
        self.plot_peak_lines=plot_stems(self,update=True,color='k')
        # update information window
        update_text(self.text_snr, 'SNR: '+str(self.SNR))
        update_text(self.text_kernel, 'Kernel size: '+str(self.kernelsize))
        # if dspec returns 0 components - display a 0 component fit
        if self.dsp.ncomps!=0:
            Decomposer.fit_spectrum_with_guesses(self.decomposer,self.guesses,fittype=self.fittype)
        # get the model
        self.modeldict=get_model_info(self)
        # recreate the model
        self.mod,self.res,self.totmod=recreate_model(self)
        # plot the model
        update_plot_model(self,update=True)
        # update the information window
        update_text(self.text_ncomp, 'number of components: '+str(self.modeldict['ncomps']))
        print_fit_information(self)

        # update plot
        self.fig.canvas.draw()

    def update_kernelsize(self,pos=None):
        """
        This controls what happens if the kernelsize slider is updated

        Parameters:
        -----------
        pos : position on the slider

        """
        # new kernel size
        self.kernelsize=int(round(pos))
        #compute new dsp
        self.dsp = compute_dsp(self)
        # update spectrum plot
        self.plot_smooth=plot_spectrum(self,self.specx,self.ysmooth,update=True,plottoupdate=self.plot_smooth)
        self.plot_peak_markers=plot_peak_locations(self,update=True,plottoupdate=self.plot_peak_markers)
        self.plot_peak_lines=plot_stems(self,update=True,color='k')
        # update deriv plot
        ymin=np.min([np.min(self.d1/np.max(self.d1)),np.min(self.d2/np.max(self.d2)),np.min(self.d3/np.max(self.d3)),np.min(self.d4/np.max(self.d4))])
        ymax=np.max([np.max(self.d1/np.max(self.d1)),np.max(self.d2/np.max(self.d2)),np.max(self.d3/np.max(self.d3)),np.max(self.d4/np.max(self.d4))])
        lim=np.max(np.abs([ymin,ymax]))
        plot_derivatives(self,update=True,ymin=-1*lim,ymax=lim)
        # update information window
        update_text(self.text_snr, 'SNR: '+str(self.SNR))
        update_text(self.text_kernel, 'Kernel size: '+str(self.kernelsize))
        # if dspec returns 0 components - display a 0 component fit
        if self.dsp.ncomps!=0:
            Decomposer.fit_spectrum_with_guesses(self.decomposer,self.guesses,fittype=self.fittype)
        # get the model
        self.modeldict=get_model_info(self)
        # recreate the model
        self.mod,self.res,self.totmod=recreate_model(self)
        # plot the model
        update_plot_model(self,update=True)
        # update the information window
        update_text(self.text_ncomp, 'number of components: '+str(self.modeldict['ncomps']))
        print_fit_information(self)

        # update plot
        self.fig.canvas.draw()

    def setup_legend_connections(self, legend, lookup_artist,lookup_handle):
        """
        setting up the connections for the interactive plot legend

        Parameters
        ----------
        legend : matplotlib legend
        lookup_artist : matplotlib artist
        lookup_handle : matplotlib legend handles

        """
        for artist in legend.texts + legend.legendHandles:
            artist.set_picker(10) # 10 points tolerance

        self.fig.canvas.mpl_connect('pick_event', lambda event: self.on_pick_legend(event, lookup_artist, lookup_handle) )
        self.fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click_legend(event, lookup_artist, lookup_handle))

    def build_legend_lookups(self, legend):
        """
        creates the lookup values for the interactive plot legend

        Parameters
        ----------
        legend : matplotlib legend

        """
        labels = [t.get_text() for t in legend.texts]
        handles = legend.legendHandles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))
        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist
        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))

        return lookup_artist, lookup_handle

    def on_pick_legend(self, event, lookup_artist, lookup_handle):
        """
        this controls what happens what happens when the legend is selected

        Parameters:
        -----------
        event : pick event
        lookup_artist : matplotlib artist
        lookup_handle : matplotlib legend handles

        """
        handle = event.artist
        if handle in lookup_artist:
            artist = lookup_artist[handle]
            artist.set_visible(not artist.get_visible())
            self.update_legend(lookup_artist,lookup_handle)

    def on_click_legend(self, event, lookup_artist, lookup_handle):
        """
        show all or hide all

        Parameters:
        -----------
        event : pick event
        lookup_artist : matplotlib artist
        lookup_handle : matplotlib legend handles

        """
        if event.button == 3:
            visible = False
        elif event.button == 2:
            visible = True
        else:
            return
        for artist in lookup_artist.values():
            artist.set_visible(visible)
        self.update_legend(lookup_artist,lookup_handle)

    def update_legend(self, lookup_artist, lookup_handle):
        """
        This controls updating the legend

        Parameters:
        -----------
        lookup_artist : matplotlib artist
        lookup_handle : matplotlib legend handles

        """
        for artist in lookup_artist.values():
            handle = lookup_handle[artist]
            if artist.get_visible():
                handle.set_visible(True)
            else:
                handle.set_visible(False)
        self.fig.canvas.draw()

    def toggle_residuals(self,event):
        """
        Toggles the residuals on and off
        """
        self.plot_res.set_visible(not self.plot_res.get_visible())
        self.fig.canvas.draw()

    def toggle_smooth(self,event):
        """
        Toggles the smoothed spectrum on and off
        """
        self.plot_smooth.set_visible(not self.plot_smooth.get_visible())
        self.fig.canvas.draw()

    def toggle_total(self,event):
        """
        Toggles the summed model (over all components) on and off
        """
        if np.size(self.plot_model)!=0:
            for i in range(len(self.plot_model)):
                plot=self.plot_model[i][0]
                self.plot_model[i][0].set_visible(not self.plot_model[i][0].get_visible())
                self.fig.canvas.draw()
        self.plot_tot.set_visible(not self.plot_tot.get_visible())
        self.fig.canvas.draw()

def retrieve_spectrum(self,spectrum_index,index):
    """
    Retrieves the spectrum
    """
    if self.method=='scouse':
        if self.scouse_stage_6:
            return self.fit_dict[self.spectra[index]]
        else:
            return self.fit_dict[self.parent[index]][self.spectra[index]]
    else:
        pass

def get_spectral_info(self):
    """
    Return the channel values
    """
    if self.method=='scouse':
        self.specx=self.scouseobject.xtrim
        self.specy=self.my_spectrum.spectrum[self.scouseobject.trimids]
        self.specrms=self.my_spectrum.rms
    else:
        self.specx = self.individual[self.index,0,:]
        self.specy = self.individual[self.index,1,:]
        self.specrms = calc_rms(self.individual[self.index,1,:])

def calc_rms(spectrum):
    """
    Returns the spectral rms

    Parameters
    ----------
    Spectrum : ndarray
        An individual spectrum

    """
    from astropy.stats import median_absolute_deviation
    # Find all negative values
    negative_indices = (spectrum < 0.0)
    spectrum_negative_values = spectrum[negative_indices]
    reflected_noise = np.concatenate((spectrum[negative_indices],
                                               abs(spectrum[negative_indices])))
    # Compute the median absolute deviation
    MAD = median_absolute_deviation(reflected_noise)
    # For pure noise you should have roughly half the spectrum negative. If
    # it isn't then you need to be a bit more conservative
    if len(spectrum_negative_values) < 0.47*len(spectrum):
        maximum_value = 3.5*MAD
    else:
        maximum_value = 4.0*MAD
    noise = spectrum[spectrum < abs(maximum_value)]
    rms = np.sqrt(np.sum(noise**2) / np.size(noise))

    return rms

def update_index(self,_type):
    """
    Updates the index for the navigator
    """
    if _type=='next':
        if self.index==len(self.spectra)-1:
            value = len(self.spectra)-1
        else:
            value = self.index+1
    elif _type=='previous':
        if self.index==0:
            value = 0
        else:
            value = self.index-1
    elif _type=='start':
        value = 0
    elif _type=='end':
        value = len(self.spectra)-1
    else:
        pass
    return value

def compute_dsp(self):
    """
    Computes derivative spectroscopy and sets some global values
    """
    from scousepy.dspec import DSpec
    dsp = DSpec(self.specx,self.specy,self.specrms,SNR=self.SNR,kernelsize=self.kernelsize)
    self.ysmooth = dsp.ysmooth
    self.d1 = dsp.d1
    self.d2 = dsp.d2
    self.d3 = dsp.d3
    self.d4 = dsp.d4
    self.ncomps = dsp.ncomps
    self.peaks = dsp.peaks
    self.centroids = dsp.centroids
    self.widths = dsp.widths
    self.guesses =dsp.guesses
    return dsp

def recreate_model(self):
    """
    Recreates model from parameters in modeldict

    """
    import pyspeckit
    # Make pyspeckit be quiet
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        old_log = log.level
        log.setLevel('ERROR')
        # generate a spectrum
        if self.modeldict['ncomps'] != 0.0:
            mod = np.zeros([len(self.specx), int(self.modeldict['ncomps'])])
            for k in range(int(self.modeldict['ncomps'])):
                modparams = self.modeldict['params'][(k*len(self.modeldict['parnames'])):(k*len(self.modeldict['parnames']))+len(self.modeldict['parnames'])]
                mod[:,k] = self.spectrum.specfit.get_model_frompars(self.specx, modparams)
            totmod = np.nansum(mod, axis=1)
            res = self.specy-totmod
        else:
            mod = np.zeros([len(self.specx), 1])
            totmod = np.zeros([len(self.specx), 1])
            res = self.specy
        log.setLevel(old_log)

    return mod, res, totmod

def get_model_info(self):
    """
    Framework for model solution dictionary
    """
    if self.decomposer.modeldict is not None:
        modeldict=self.decomposer.modeldict
    else:
        modeldict={}
        modeldict['fittype']=None
        modeldict['parnames']=['amplitude','shift','width']
        modeldict['ncomps']=0
        modeldict['params']=[0.0,0.0,0.0]
        modeldict['errors']=[0.0,0.0,0.0]
        modeldict['rms']=self.spectrum.error[0]
        modeldict['residstd']= np.std(self.spectrum.data)
        modeldict['chisq']=0.0
        modeldict['dof']=0.0
        modeldict['redchisq']=0.0
        modeldict['AIC']=0.0
        modeldict['fitconverge']=False
        modeldict['method']=self.decomposer.method

    modeldict['SNR']=self.SNR
    modeldict['kernelsize']=self.kernelsize

    return modeldict

def get_aic(self):
    """
    Computes the AIC value
    """
    from astropy.stats import akaike_info_criterion as aic
    logl = self.spectrum.specfit.fitter.logp(self.spectrum.xarr, self.spectrum.data, self.spectrum.error)
    return aic(logl, int(self.spectrum.specfit.npeaks)+(int(self.spectrum.specfit.npeaks)*3.), len(self.spectrum.xarr))

def setup_plot_window(self,ax,ymin=None,ymax=None):
    """
    GUI setup
    """
    window=self.fig.add_axes(ax)
    window.tick_params(axis='x', which='major', length=5, direction='in',pad=5)
    window.tick_params(axis='x', which='minor', length=3, direction='in')
    window.tick_params(axis='y', which='major', length=5, direction='in',pad=5, rotation=90)
    window.tick_params(axis='y', which='minor', length=3, direction='in')
    window.grid(color='grey', alpha=0.4, linestyle=':', which='major')
    window.grid(color='grey', alpha=0.1, linestyle=':', which='minor')
    window.set_facecolor('whitesmoke')
    # set the axis limits
    window.set_ylim([ymin,ymax])
    return window

def setup_information_window(self):
    """
    GUI setup
    """
    self.information_window=self.fig.add_axes(self.information_window_ax)
    self.information_window.set_xticklabels([])
    self.information_window.set_yticklabels([])
    self.information_window.set_xticks([])
    self.information_window.set_yticks([])
    self.information_window.text(0.01, 0.92, 'derivative spectroscopy information:',
                                 transform=self.information_window.transAxes,
                                 fontsize=10, fontweight='bold')

def plot_spectrum(self,x,y,update=False,plottoupdate=None,**kwargs):
    """
    GUI setup
    """
    if update:
        self.spectrum_window.set_xlim(np.nanmin(x),np.nanmax(x))
        plottoupdate.set_xdata(x)
        plottoupdate.set_ydata(y)
        return plottoupdate
    else:
        self.spectrum_window.set_xlim(np.nanmin(x),np.nanmax(x))
        plot=self.spectrum_window.plot(self.specx,y,drawstyle='steps',**kwargs)
        return plot

def plot_snr(self,**kwargs):
    """
    GUI setup
    """
    return self.spectrum_window.plot([np.min(self.specx),np.max(self.specx)],[self.SNR*self.specrms,self.SNR*self.specrms],**kwargs)

def plot_peak_locations(self,update=False,plottoupdate=None,**kwargs):
    """
    GUI setup
    """
    if update:
        plottoupdate.set_xdata(self.centroids)
        plottoupdate.set_ydata(self.peaks)
        return plottoupdate
    else:
        return self.spectrum_window.plot(self.centroids,self.peaks,**kwargs, label='dspec prediction')

def plot_stems(self,update=False,**kwargs):
    """
    GUI setup
    """
    if update:
        for i in range(len(self.plot_peak_lines)):
            self.plot_peak_lines[i].pop(0).remove()

    self.plot_peak_lines=[]
    for i in range(self.ncomps):
        self.plot_peak_lines_indiv = self.spectrum_window.plot([self.centroids[i], self.centroids[i]],[0,self.peaks[i]],**kwargs)
        self.plot_peak_lines.append(self.plot_peak_lines_indiv)
    return self.plot_peak_lines

def plot_derivatives(self,update=False,ymin=None,ymax=None):
    """
    GUI setup
    """
    import numpy as np

    d1=self.d1/np.max(self.d1)
    d2=self.d2/np.max(self.d2)
    d3=self.d3/np.max(self.d3)
    d4=self.d4/np.max(self.d4)
    # plot the data
    if update:
        self.deriv_window.set_ylim([ymin,ymax])
        self.plot_d1.set_ydata(d1)
        self.plot_d2.set_ydata(d2)
        self.plot_d3.set_ydata(d3)
        self.plot_d4.set_ydata(d4)
    else:
        self.plot_d1,=self.deriv_window.plot(self.specx,d1,lw=0.5,label='f$^{\prime}$(x)')
        self.plot_d2,=self.deriv_window.plot(self.specx,d2,lw=2.0,label='f$^{\prime\prime}$(x)')
        self.plot_d3,=self.deriv_window.plot(self.specx,d3,lw=0.5,label='f$^{\prime\prime\prime}$(x)')
        self.plot_d4,=self.deriv_window.plot(self.specx,d4,lw=0.5,label='f$^{\prime\prime\prime\prime}$(x)')

def plot_residuals(self,y,residkwargs):
    """
    GUI setup
    """
    return self.spectrum_window.plot(self.specx,y,drawstyle='steps',label='residual',**residkwargs)

def plot_model(self,y,label,modelkwargs):
    """
    GUI setup
    """
    return self.spectrum_window.plot(self.specx,y,label=label,**modelkwargs)

def update_plot_model(self,update=False):
    """
    GUI setup
    """
    if self.dsp.ncomps < 10:
        update_text(self.text_fitinformation,'pyspeckit fit information: ')

        if self.modeldict['fitconverge']:
            update_text(self.text_converge,"Fit has converged...", color='limegreen')
        else:
            update_text(self.text_converge,"Fit has not converged...try increasing the SNR and/or the kernel size.", color='red')

        if update:
            self.plot_res.remove()
            self.plot_tot.remove()
            if np.size(self.plot_model)!=0:
                for i in range(len(self.plot_model)):
                    self.plot_model[i].pop(0).remove()

        if not self.modeldict['fitconverge']:
            self.plot_res,=plot_residuals(self,self.specy,self.residkwargs)
            self.plot_tot,=plot_model(self,np.zeros_like(self.specy),'total model',self.totmodkwargs)
            self.plot_model=[]
        else:
            #plot residuals
            self.plot_res,=plot_residuals(self,self.res,self.residkwargs)
            self.plot_tot,=plot_model(self,self.totmod,'total model',self.totmodkwargs)
            # now overplot the model
            self.plot_model=[]
            for k in range(int(self.modeldict['ncomps'])):
                self.plot_model_indiv=plot_model(self,self.mod[:,k],'comp '+str(k),self.modelkwargs)
                self.plot_model.append(self.plot_model_indiv)
    else:
        update_text(self.text_fitinformation,'pyspeckit fit information: >10 components detected, autofitting may be slow. Use "fit (dspec)" button to fit')

    # plot a legend
    self.spectrum_legend = self.spectrum_window.legend(loc=2,frameon=False,fontsize=8)
    self.spectrum_window_lookup_artist, self.spectrum_window_lookup_handle = self.build_legend_lookups(self.spectrum_legend)
    self.setup_legend_connections(self.spectrum_legend, self.spectrum_window_lookup_artist, self.spectrum_window_lookup_handle)
    self.update_legend(self.spectrum_window_lookup_artist, self.spectrum_window_lookup_handle)

def print_information(self,xloc,yloc,str,**kwargs):
    """
    GUI setup
    """
    return self.information_window.text(xloc,yloc,str,transform=self.information_window.transAxes, **kwargs)

def print_fit_information(self):
    """
    GUI setup
    """
    strchisq=str(("chisq: {0}").format(np.around(self.modeldict['chisq'],decimals=2)))
    strredchisq=str(("red chisq: {0}").format(np.around(self.modeldict['redchisq'],decimals=2)))
    straic=str(("AIC: {0}").format(np.around(self.modeldict['AIC'],decimals=2)))
    update_text(self.text_chisq,strchisq)
    update_text(self.text_redchisq,strredchisq)
    update_text(self.text_aic,straic)

    if self.modeldict['ncomps']==0:
        strpeaks=str(("{0}:  {1} +/- {2}").format(self.modeldict['parnames'][0],np.around(self.modeldict['params'][0], decimals=2),np.around(self.modeldict['errors'][0],decimals=2)))
        strcentroids=str(("{0}:  {1} +/- {2}").format(self.modeldict['parnames'][1],np.around(self.modeldict['params'][1], decimals=2),np.around(self.modeldict['errors'][1],decimals=2)))
        strwidths=str(("{0}:  {1} +/- {2}").format(self.modeldict['parnames'][2],np.around(self.modeldict['params'][2], decimals=2),np.around(self.modeldict['errors'][2],decimals=2)))
        update_text(self.text_peaks,strpeaks)
        update_text(self.text_centroids,strcentroids)
        update_text(self.text_widths,strwidths)
    else:
        parrange=[]
        for i in range(self.modeldict['ncomps']):
            parlow = int((i*len(self.modeldict['parnames'])))
            parhigh = int((i*len(self.modeldict['parnames']))+len(self.modeldict['parnames']))
            _parrange = np.arange(parlow,parhigh)
            parrange.append(_parrange)
        parrange=np.asarray(parrange,dtype='int').T

        textobjects=[self.text_peaks,self.text_centroids,self.text_widths]
        for i in range(0, len(self.modeldict['parnames'])):
            _parrange=parrange[i]
            for j in range(self.modeldict['ncomps']):
                if j==0:
                    mystring=str(("{0}: {1} +/- {2}").format(self.modeldict['parnames'][i],np.around(self.modeldict['params'][_parrange[j]], decimals=2),np.around(self.modeldict['errors'][_parrange[j]],decimals=2)))
                else:
                    mystring+=str((";   {0} +/- {1}").format(np.around(self.modeldict['params'][_parrange[j]], decimals=2),np.around(self.modeldict['errors'][_parrange[j]],decimals=2)))
            update_text(textobjects[i],mystring)

def print_to_file(self):
    """
    Prints best-fitting solutions to file
    """
    # imports
    from astropy.io import ascii
    from astropy.table import Table
    from astropy.table import Column
    # create a table
    table = Table(meta={'name': 'best-fitting model solutions'})

    solnlist = []
    for key in self.modelstore.keys():
        headings=get_headings(self, self.modelstore[key])
        for i in range(self.modelstore[key]['ncomps']):
            solution=get_soln_desc(key,i,self.modelstore[key])
            solnlist.append(solution)

    solnarr = np.asarray(solnlist).T

    for j in range(len(solnarr[:,0])):
        table[headings[j]] = Column(solnarr[j,:])

    table.write(self.outputfile, format='ascii', overwrite=True, delimiter='\t')

def get_headings(self, dict):
    """
    Table headings for output

    Notes:

    This is awful but it works.
    """

    cont = True
    keys = list(dict.keys())
    count = 0
    headings = []
    # find the first spectral averaging area where there is a fit
    # and get the parameter names
    headings_non_specific = ['index', 'ncomps', 'rms', 'residstd',
                             'chisq','dof', 'redchisq', 'AIC',
                             'fitconverge','SNR', 'kernelsize',
                             'method' ]
    #These ones depend on the model used by pyspeckit
    headings_params = dict['parnames']
    headings_errs = [str('err {0}').format(dict['parnames'][k]) for k in range(len(dict['parnames']))]
    # This is messy
    headings_pars = [[headings_params[k], headings_errs[k]] for k in range(len(dict['parnames']))]
    headings_pars = [par for pars in headings_pars for par in pars]
    headings = headings_non_specific[0:2]+headings_pars+headings_non_specific[2::]

    return headings

def get_soln_desc(key,idx,dict):
    """
    Returns the solution in the format:

    ncomps, param1, err1, .... paramn, errn, rms, residstd, chi2, dof, chi2red,
    aic, fitconverge, snr, kernelsize, method
    """
    params_non_specific = [key, dict['ncomps'], dict['rms'],
                           dict['residstd'],dict['chisq'],dict['dof'],
                           dict['redchisq'], dict['AIC'], dict['fitconverge'],
                           dict['SNR'], dict['kernelsize'],dict['method'] ]

    parlow = int((idx*len(dict['parnames'])))
    parhigh = int((idx*len(dict['parnames']))+len(dict['parnames']))
    parrange = np.arange(parlow,parhigh)

    paramarr = np.array(dict['params'])
    errarr = np.array(dict['errors'])
    params = paramarr[parrange]
    errors = errarr[parrange]
    # This is messy
    parameters = [[params[j], errors[j]] for j in range(len(dict['parnames']))]
    parameters = [par for pars in parameters for par in pars]
    solution_desc = params_non_specific[0:2]+parameters+params_non_specific[2::]

    return solution_desc

def update_text(textobject,textstring,color=None):
    """
    GUI setup
    """
    textobject.set_text(textstring)
    if color is not None:
        textobject.set_color(color)

def make_slider(ax,name,min,max,function,**kwargs):
    """
    GUI setup
    """
    from matplotlib.widgets import Slider
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    myslider=Slider(ax,name,min,max,**kwargs)
    myslider.on_changed(function)
    myslider.drawon = False

    return myslider

def make_button(ax,name,function,**kwargs):
    """
    GUI setup
    """
    from matplotlib.widgets import Button
    mybutton=Button(ax,name,**kwargs)
    mybutton.on_clicked(function)
    return mybutton

def make_textbox(ax,text,function,**kwargs):
    """
    GUI setup
    """
    from matplotlib.widgets import TextBox
    mytextbox=TextBox(ax,'',initial=text, color='1')
    mytextbox.on_submit(function)
    return mytextbox
