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

class ScouseFitter(object):

    def __init__(self, method='scouse', scouseobject=None, scouse_single=False,
                 spectra=None, spectra_dict=None, fitrange=-1,
                 x=None, y=None, rms=None):

        """
        """

        self.method=method
        self.scouseobject=scouseobject
        self.scouse_single=scouse_single
        self.spectra=spectra
        self.spectra_dict=spectra_dict
        self.fitrange=fitrange
        self.models={}

        import matplotlib.pyplot as plt
        plt.ioff()
        from scousepy.dspec import DSpec

        if method=='scouse':
            # set some defaults for pyspeckit
            import astropy.units as u
            self.xarrkwargs={'unit':'km/s',
                             'refX': self.scouseobject.cube.wcs.wcs.restfrq*u.Hz,
                             'velocity_convention': 'radio'}
            self.unit=self.scouseobject.cube.header['BUNIT']

            if scouse_single:
                pass
            else:
                if (scouseobject is None) or (spectra is None):
                    # make sure that the scouse object has been sent
                    ValueError("Please include both the scousepy object and the spectra to be fit")
                else:
                    # if it has then establish the fit range
                    if fitrange==-1: # fit all
                        # Create a list of indices so that we can find the relevant
                        # spectra
                        self.indexlist=np.arange(0,int(np.size(self.spectra)))
                        # index of the first spectrum to be fit
                        self.index=0
                        # lets retrieve the spectrum
                        self.my_spectrum=retrieve_spectrum(self,self.spectra[self.index])
                        get_spectral_info(self)
                        self.spectrum=generate_pyspeckit_spectrum(self, xarrkwargs=self.xarrkwargs,unit=self.unit)

        elif method=='cube':
            self.xarrkwargs={}
            self.unit={}
        elif method=='single':
            self.xarrkwargs={}
            self.unit={}
            assert x is not None, "To create a spectrum provide the x axis"
            assert y is not None, "To create a spectrum provide the y axis"
            assert rms is not None, "To create a spectrum provide the rms"
            self.get_spectral_info(method=method,x=x,y=y,rms=rms)
            self.spectrum = generate_template_spectrum(self)
        else:
            ValueError("Please use a valid method type: 'scouse', 'cube', 'single'")

        # set the default values for the SNR and kernel size as well as ranges for the sliders
        self.SNR = 3
        self.minSNR = 1
        self.maxSNR = 30
        self.kernelsize = 5
        self.minkernel = 1
        self.maxkernel = 30

        # compute derivative spec
        self.dsp = compute_dsp(self)

        # initiate the plot window
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
        # plot the spectra
        self.plot_spectrum,=plot_spectrum(self,self.specy,label='spec')
        self.plot_smooth,=plot_spectrum(self,self.ysmooth,label='smoothed spec',lw=1.5,ls=':',color='k')
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
        self.information_window_ax=[0.05, 0.08, 0.90, 0.35]
        setup_information_window(self)
        self.text_snr=print_information(self,0.01,0.84,'SNR: '+str(self.SNR), fontsize=10)
        self.text_kernel=print_information(self,0.01,0.76,'kernel size: '+str(self.kernelsize),fontsize=10)
        self.text_fitinformation=print_information(self,0.01,0.68,'pyspeckit fit information: ', fontsize=10, fontweight='bold')
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

        self.spectrum=fit_spectrum(self)
        self.modeldict=get_model_info(self)
        self.mod,self.res,self.totmod=recreate_model(self)
        update_plot_model(self)
        print_fit_information(self)

        #=======#
        # buttons
        #=======#
        self.button_previous_ax=self.fig.add_axes([0.85, 0.85, 0.1, 0.05])
        self.button_previous=make_button(self.button_previous_ax,"previous",lambda event: self.new_spectrum(event, type='previous'))

        self.button_next_ax=self.fig.add_axes([0.85, 0.775, 0.1, 0.05])
        self.button_next=make_button(self.button_next_ax,"next",lambda event: self.new_spectrum(event, type='next'))

        self.button_fit_dspec_ax=self.fig.add_axes([0.85, 0.7, 0.1, 0.05])
        self.button_fit_dspec=make_button(self.button_fit_dspec_ax,"fit (dspec)",self.dspec_manual)

        self.button_fit_manual_ax=self.fig.add_axes([0.85, 0.625, 0.1, 0.05])
        self.button_fit_manual=make_button(self.button_fit_manual_ax,"fit (manual)",self.open_scousefitter_manual)

        self.button_applyall_ax=self.fig.add_axes([0.85, 0.55, 0.1, 0.05])
        self.button_applyall=make_button(self.button_applyall_ax,"apply dspec to all",self.dspec_apply_to_all)

        self.button_stop_ax=self.fig.add_axes([0.85, 0.475, 0.1, 0.05])
        self.button_stop=make_button(self.button_stop_ax,"stop",self.toggle_residuals)

        #=======#
        # sliders
        #=======#
        self.slider_snr_ax=self.fig.add_axes([0.0875, 0.91, 0.3, 0.02])
        self.slider_snr=make_slider(self.slider_snr_ax,"SNR",self.minSNR,self.maxSNR,self.update_SNR,valinit=self.SNR, valfmt="%i")

        self.slider_kernel_ax=self.fig.add_axes([0.4875, 0.91, 0.3, 0.02])
        self.slider_kernel=make_slider(self.slider_kernel_ax,"kernel",self.minkernel,self.maxkernel,self.update_kernelsize,valinit=self.kernelsize, valfmt="%i")

    def show(self):
        import matplotlib.pyplot as plt
        plt.show()

    def open_scousefitter_manual(self, event):
        """
        This controls the manual fitting if no reasonable solution can be found
        with the derivative spectroscopy method

        """
        from scousepy.scousefittermanual import ScouseFitterManual
        ManualFitter = ScouseFitterManual(self)
        fit = ManualFitter.manualfit()
        self.spectrum=fit.spectrum
        # fit and plot
        self.modeldict=get_model_info(self)
        self.mod,self.res,self.totmod=recreate_model(self)
        update_plot_model(self,update=True)
        update_text(self.text_ncomp, 'number of components: '+str(self.modeldict['ncomps']))
        print_fit_information(self)

        # update plot
        self.fig.canvas.draw()

    def new_spectrum(self,event,type=None):
        """
        This descibes what happens when a new spectrum is selected using the
        buttons "previous" and "next"

        Parameters
        ----------
        event : button click event
        type : describes what button has been pressed

        """
        if self.modeldict is not None:
            # always save the current model if we move on
            self.models[self.index]=self.modeldict
        else:
            self.modeldict=get_model_info(self)
            print(self.modeldict)

        # retrieve the new index
        self.index=update_index(self,type)
        # get the relevant spectrum
        self.my_spectrum=retrieve_spectrum(self,self.spectra[self.index])
        # get the spectral information
        get_spectral_info(self)
        # update the spectrum
        self.spectrum=generate_pyspeckit_spectrum(self, xarrkwargs=self.xarrkwargs,unit=self.unit)
        #compute new dsp
        self.dsp = compute_dsp(self)

        # update spectrum plot
        ymax=np.max([self.SNR*self.specrms, np.max(self.specy)])+\
             0.2*np.max([self.SNR*self.specrms, np.max(self.specy)])
        ymin=np.min(self.specy)-0.2*np.max(self.specy)
        self.spectrum_window.set_ylim([ymin,ymax])
        self.plot_spectrum=plot_spectrum(self,self.specy,update=True,plottoupdate=self.plot_spectrum)
        self.plot_smooth=plot_spectrum(self,self.ysmooth,update=True,plottoupdate=self.plot_smooth)
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

        # check to see if there a model already exists
        if self.index in self.models.keys():
            self.spectrum=fit_spectrum(self)
            self.modeldict=self.models[self.index]
            self.mod,self.res,self.totmod=recreate_model(self)
            update_plot_model(self,update=True)
            update_text(self.text_ncomp, 'number of components: '+str(self.modeldict['ncomps']))
            print_fit_information(self)
        else:
            if self.dsp.ncomps!=0:
                # fit and plot
                self.spectrum=fit_spectrum(self)
            self.modeldict=get_model_info(self)
            self.mod,self.res,self.totmod=recreate_model(self)
            update_plot_model(self,update=True)
            update_text(self.text_ncomp, 'number of components: '+str(self.modeldict['ncomps']))
            print_fit_information(self)

        # update plot
        self.fig.canvas.draw()

    def dspec_manual(self, event):
        """
        This controls the manual dspec fitter.

        """
        # update the spectrum
        self.spectrum=generate_pyspeckit_spectrum(self, xarrkwargs=self.xarrkwargs,unit=self.unit)
        #compute new dsp
        self.dsp = compute_dsp(self)
        # update spectrum plot
        self.plot_peak_markers=plot_peak_locations(self,update=True,plottoupdate=self.plot_peak_markers)
        self.plot_peak_lines=plot_stems(self,update=True,color='k')
        # fit and plot
        self.spectrum=fit_spectrum(self)
        self.modeldict=get_model_info(self)
        self.mod,self.res,self.totmod=recreate_model(self)
        update_plot_model(self,update=True)
        update_text(self.text_ncomp, 'number of components: '+str(self.modeldict['ncomps']))
        print_fit_information(self)

        # update plot
        self.fig.canvas.draw()

    def dspec_apply_to_all(self,event):
        """
        controls what happens if the apply dspec to all button is pressed - will
        save the current slider values apply these to all spectra and save
        """
        for i in self.indexlist:
            # retrieve the new index
            index=int(i)
            if not index in self.models.keys():
                # get the relevant spectrum
                self.my_spectrum=retrieve_spectrum(self,self.spectra[index])
                # get the spectral information
                get_spectral_info(self)
                # update the spectrum
                self.spectrum=generate_pyspeckit_spectrum(self, xarrkwargs=self.xarrkwargs,unit=self.unit)
                #compute new dsp
                self.dsp = compute_dsp(self)
                self.spectrum=fit_spectrum(self)
                self.modeldict=get_model_info(self)
                self.models[index]=self.modeldict

        self.modeldict=None

    def update_SNR(self,pos=None):
        """
        This controls what happens if the SNR slider is updated

        Parameters:
        -----------
        pos : position on the slider

        """
        # New SNR
        self.SNR = int(round(pos))
        # update the spectrum
        self.spectrum=generate_pyspeckit_spectrum(self, xarrkwargs=self.xarrkwargs,unit=self.unit)
        #compute new dsp
        self.dsp = compute_dsp(self)

        # update spectrum plot
        ymax=np.max([self.SNR*self.specrms, np.max(self.specy)])+\
             0.2*np.max([self.SNR*self.specrms, np.max(self.specy)])
        ymin=np.min(self.specy)-0.2*np.max(self.specy)
        self.spectrum_window.set_ylim([ymin,ymax])
        self.plot_SNR.set_ydata([self.SNR*self.specrms,self.SNR*self.specrms])
        self.plot_peak_markers=plot_peak_locations(self,update=True,plottoupdate=self.plot_peak_markers)
        self.plot_peak_lines=plot_stems(self,update=True,color='k')
        # update information window
        update_text(self.text_snr, 'SNR: '+str(self.SNR))

        if self.dsp.ncomps!=0:
            # fit and plot
            self.spectrum=fit_spectrum(self)

        self.modeldict=get_model_info(self)
        self.mod,self.res,self.totmod=recreate_model(self)
        update_plot_model(self,update=True)
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
        self.plot_smooth=plot_spectrum(self,self.ysmooth,update=True,plottoupdate=self.plot_smooth)
        self.plot_peak_markers=plot_peak_locations(self,update=True,plottoupdate=self.plot_peak_markers)
        self.plot_peak_lines=plot_stems(self,update=True,color='k')
        # update deriv plot
        ymin=np.min([np.min(self.d1/np.max(self.d1)),np.min(self.d2/np.max(self.d2)),np.min(self.d3/np.max(self.d3)),np.min(self.d4/np.max(self.d4))])
        ymax=np.max([np.max(self.d1/np.max(self.d1)),np.max(self.d2/np.max(self.d2)),np.max(self.d3/np.max(self.d3)),np.max(self.d4/np.max(self.d4))])
        lim=np.max(np.abs([ymin,ymax]))
        plot_derivatives(self,update=True,ymin=-1*lim,ymax=lim)
        # update information window
        update_text(self.text_snr, 'SNR: '+str(self.SNR))

        if self.dsp.ncomps!=0:
            # fit and plot
            self.spectrum=fit_spectrum(self)

        self.modeldict=get_model_info(self)
        self.mod,self.res,self.totmod=recreate_model(self)
        update_plot_model(self,update=True)
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
        self.plot_res.set_visible(not self.plot_res.get_visible())
        self.fig.canvas.draw()

    def toggle_smooth(self,event):
        self.plot_smooth.set_visible(not self.plot_smooth.get_visible())
        self.fig.canvas.draw()

    def toggle_total(self,event):
        if np.size(self.plot_model)!=0:
            for i in range(len(self.plot_model)):
                plot=self.plot_model[i][0]
                self.plot_model[i][0].set_visible(not self.plot_model[i][0].get_visible())
                self.fig.canvas.draw()
        self.plot_tot.set_visible(not self.plot_tot.get_visible())
        self.fig.canvas.draw()

def retrieve_spectrum(self,index):
    if self.method=='scouse':
        return self.spectra_dict[index]
    else:
        pass

def get_spectral_info(self,x=None,y=None,rms=None):
    if self.method=='scouse':
        self.specx=self.scouseobject.xtrim
        self.specy=self.my_spectrum.ytrim
        self.specrms=self.my_spectrum.rms
    else:
        self.specx = x
        self.specy = y
        self.specrms = rms

def generate_pyspeckit_spectrum(self,plotkwargs={},xarrkwargs={},
                                unit=None,doplot=False):
    import pyspeckit
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        old_log = log.level
        log.setLevel('ERROR')
        spectrum=pyspeckit.Spectrum(data=self.specy,
                                    error=np.ones(len(self.specy))*self.specrms,
                                    xarr=self.specx,
                                    doplot=doplot,
                                    plotkwargs=plotkwargs,
                                    unit=unit,
                                    xarrkwargs=xarrkwargs
                                    )
    log.setLevel(old_log)
    return spectrum

def update_index(self,type):
    if type=='next':
        if self.index==len(self.spectra)-1:
            pass
        else:
            self.index+=1
    elif type=='previous':
        if self.index==0:
            pass
        else:
            self.index-=1
    else:
        pass
    return self.index

def compute_dsp(self):
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

def generate_template_spectrum(self):
    """
    Generate a template spectrum to be passed to the fitter.

    """
    import pyspeckit
    import astropy.units as u
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        old_log = log.level
        log.setLevel('ERROR')
        template_spectrum = pyspeckit.Spectrum(data=self.specy,
                                               error=np.ones_like(self.specy)*self.specrms,
                                               xarr=self.specx,
                                               doplot=False,
                                               verbose=False,
                                               )
        log.setLevel(old_log)
    return template_spectrum

def fit_spectrum(self):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        old_log = log.level
        log.setLevel('ERROR')
        #parvalues=np.asarray(self.guesses)
        #parlimited=[(True, False), (False, False), (False, False)]
        #parlimits=[(0,0),(0,0),(0,0)]
        #parinfo={'parvalues':parvalues, 'parlimited':parlimited, 'parlimits':parlimits}
        # self.spectrum.specfit(interactive=False,
        #                       clear_all_connections=True,
        #                       xmin=np.min(self.specx),
        #                       xmax=np.max(self.specx),
        #                       fittype = 'gaussian',
        #                       guesses = self.guesses,
        #                       minpars = [0,0,0,0,0,0],
        #                       maxpars = [0,0,0,0,0,0],
        #                       fixed = [True, False, False, False, False, False],
        #                       verbose=False,
        #                       use_lmfit=True)
        self.spectrum.specfit(interactive=False,
                              clear_all_connections=True,
                              xmin=np.min(self.specx),
                              xmax=np.max(self.specx),
                              fittype = 'gaussian',
                              guesses = self.guesses,
                              verbose=False,
                              use_lmfit=True)
        log.setLevel(old_log)
        return self.spectrum

def recreate_model(self):
    """
    Recreates model from parameters

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

def setup_plot_window(self,ax,ymin=None,ymax=None):
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
    self.information_window=self.fig.add_axes(self.information_window_ax)
    self.information_window.set_xticklabels([])
    self.information_window.set_yticklabels([])
    self.information_window.set_xticks([])
    self.information_window.set_yticks([])
    self.information_window.text(0.01, 0.92, 'derivative spectroscopy information:',
                                 transform=self.information_window.transAxes,
                                 fontsize=10, fontweight='bold')

def plot_spectrum(self,y,update=False,plottoupdate=None,**kwargs):
    if update:
        plottoupdate.set_ydata(y)
        return plottoupdate
    else:
        return self.spectrum_window.plot(self.specx,y,drawstyle='steps',**kwargs)

def plot_snr(self,**kwargs):
    return self.spectrum_window.plot([np.min(self.specx),np.max(self.specx)],[self.SNR*self.specrms,self.SNR*self.specrms],**kwargs)

def plot_peak_locations(self,update=False,plottoupdate=None,**kwargs):
    if update:
        plottoupdate.set_xdata(self.centroids)
        plottoupdate.set_ydata(self.peaks)
        return plottoupdate
    else:
        return self.spectrum_window.plot(self.centroids,self.peaks,**kwargs, label='dspec prediction')

def plot_stems(self,update=False,**kwargs):

    if update:
        for i in range(len(self.plot_peak_lines)):
            self.plot_peak_lines[i].pop(0).remove()

    self.plot_peak_lines=[]
    for i in range(self.ncomps):
        self.plot_peak_lines_indiv = self.spectrum_window.plot([self.centroids[i], self.centroids[i]],[0,self.peaks[i]],**kwargs)
        self.plot_peak_lines.append(self.plot_peak_lines_indiv)
    return self.plot_peak_lines

def plot_derivatives(self,update=False,ymin=None,ymax=None):
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
    return self.spectrum_window.plot(self.specx,y,drawstyle='steps',label='residual',**residkwargs)

def plot_model(self,y,label,modelkwargs):
    return self.spectrum_window.plot(self.specx,y,label=label,**modelkwargs)

def update_plot_model(self,update=False):
    if self.dsp.ncomps < 10:
        update_text(self.text_fitinformation,'pyspeckit fit information: ')

        if update:
            self.plot_res.remove()
            self.plot_tot.remove()
            if np.size(self.plot_model)!=0:
                for i in range(len(self.plot_model)):
                    self.plot_model[i].pop(0).remove()

        if self.dsp.ncomps == 0:
            self.plot_res,=plot_residuals(self,self.specy,self.residkwargs)
            self.plot_tot,=plot_model(self,np.zeros_like(self.specy),'total model',self.totmodkwargs)
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

def get_model_info(self):
    modeldict={}
    if self.ncomps==0:
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
        modeldict['fitconverge'] = False

    else:
        modeldict['fittype']=self.spectrum.specfit.fittype
        modeldict['parnames']=self.spectrum.specfit.fitter.parnames
        modeldict['ncomps']=int(self.spectrum.specfit.npeaks)
        modeldict['params']=self.spectrum.specfit.modelpars
        modeldict['errors']=self.spectrum.specfit.modelerrs
        modeldict['rms']=self.spectrum.error[0]
        modeldict['residstd']= np.std(self.spectrum.specfit.residuals)
        modeldict['chisq']=self.spectrum.specfit.chi2
        modeldict['dof']=self.spectrum.specfit.dof
        modeldict['redchisq']=self.spectrum.specfit.chi2/self.spectrum.specfit.dof
        modeldict['AIC']=get_aic(self)

        if None in self.spectrum.specfit.modelerrs:
            modeldict['fitconverge'] = False
            self.spectrum.specfit.modelerrs = np.zeros(len(self.spectrum.specfit.modelerrs))
        else:
            modeldict['fitconverge'] = True

    return modeldict

def get_aic(self):
    from astropy.stats import akaike_info_criterion as aic
    logl = self.spectrum.specfit.fitter.logp(self.spectrum.xarr, self.spectrum.data, self.spectrum.error)
    return aic(logl, int(self.spectrum.specfit.npeaks)+(int(self.spectrum.specfit.npeaks)*3.), len(self.spectrum.xarr))

def print_information(self,xloc,yloc,str,**kwargs):
    return self.information_window.text(xloc,yloc,str,transform=self.information_window.transAxes, **kwargs)

def print_fit_information(self):
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

def update_text(textobject,textstring):
    textobject.set_text(textstring)

def make_slider(ax,name,min,max,function,**kwargs):
    from matplotlib.widgets import Slider
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    myslider=Slider(ax,name,min,max,**kwargs)
    myslider.on_changed(function)
    myslider.drawon = False

    return myslider

def make_button(ax,name,function,**kwargs):
    from matplotlib.widgets import Button
    mybutton=Button(ax,name)
    mybutton.on_clicked(function)
    return mybutton
