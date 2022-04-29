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
import os
from scousepy.SpectralDecomposer import Decomposer
import astropy.units as u

class ScouseFitChecker(object):
    """
    Interactive model inspector for scouse

    Parameters
    ----------
    scouseobject : scouse class object
        Instance of the scouse object.

    """
    def __init__(self, scouseobject=None,
                verbose=True,
                blocksize=5.,
                selected_spectra=None,
                maps=['rms','residstd','redchisq','ncomps','AIC','chisq'],
                SNR=3,minSNR=1,maxSNR=30,
                alpha=3,minalpha=0.1,maxalpha=30,
                fittype='gaussian',
                xarrkwargs={},unit={},
                scouseobjectalt=[]):

        self.scouseobject=scouseobject
        self.scouseobjectalt=scouseobjectalt
        self.verbose=verbose
        self.check_spec_indices=[]

        # diagnostic maps
        self.maps=maps

        # related to spectral grid
        self.blocksize=int(blocksize)
        self.xpos=None
        self.ypos=None
        self.keys=None
        self.selected_spectra=selected_spectra

        # related to the individual spectrum
        self.SNR=SNR
        self.minSNR=minSNR
        self.maxSNR=maxSNR
        self.alpha=alpha
        self.minalpha=minalpha
        self.maxalpha=maxalpha

        self.speckey=None
        self.specx=None
        self.specy=None
        self.specrms=None
        self.dsp=None
        self.modeldict=None
        self.fittype=fittype
        self.xarrkwargs=xarrkwargs
        self.unit=unit

        if scouseobject is not None:
            self.xarrkwargs={'unit':'km/s',
                             'refX': self.scouseobject.cube.wcs.wcs.restfrq*u.Hz,
                             'velocity_convention': 'radio'}
            self.unit=self.scouseobject.cube.header['BUNIT']
        # imports
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib import rcParams

        rcParams['font.family']= 'Arial'
        rcParams['font.size']= 9
        rcParams['lines.linewidth']= 1.     ## line width in points
        rcParams['axes.labelsize'] =10  ## fontsize of the x any y labels
        rcParams['xtick.labelsize']=10 ## fontsize of the tick labels
        rcParams['ytick.labelsize'] =10 ## fontsize of the tick labels
        rcParams['xtick.major.pad']=4   ## distance to major tick label in points
        rcParams['ytick.major.pad']=4    ## distance to major tick label in points
        rcParams['xtick.major.size'] =4    ## major tick size in points
        rcParams['xtick.minor.size' ]=2     ## minor tick size in points
        rcParams['xtick.major.width'] =1.    ## major tick width in points
        rcParams['xtick.minor.width']=1.    ## minor tick width in points
        rcParams['ytick.major.size']= 4    ## major tick size in points
        rcParams['ytick.minor.size' ]=2      ## minor tick size in points
        rcParams['ytick.major.width']=1.    ## major tick width in points
        rcParams['ytick.minor.width']= 1.    ## minor tick width in points

        # remove some matplotlib keyboard shortcuts to prevent meltdown
        if 'q' in plt.rcParams['keymap.quit']:
            plt.rcParams['keymap.quit'].remove('q')
        if 'Q' in plt.rcParams['keymap.quit_all']:
            plt.rcParams['keymap.quit_all'].remove('Q')

        # compute diagnostics
        self.diagnostics = compute_diagnostic_plots(self)
        save_maps(self,self.diagnostics)

        #================#
        # initiate the GUI
        #================#
        self.fig = plt.figure(figsize=(16, 8))

        #===============#
        # plot window
        #===============#

        # Set up the plot defaults
        self.blank_window_ax=[0.05,0.2,0.275,0.6]
        self.blank_window=setup_plot_window(self,self.blank_window_ax)
        self.map_window=setup_map_window(self)
        self.spec_window_ax=[0.35,0.2,0.275,0.6]
        self.spec_grid_window=setup_spec_window(self)
        self.spectrum_window_ax=[0.675,0.5,0.275,0.3]
        self.spectrum_window=setup_indivspec_window(self, self.spectrum_window_ax)
        self.spectrum_window.text(0.99, 0.05, 'select legend items to toggle',
                                  transform=self.spectrum_window.transAxes,
                                  fontsize=8, ha='right')

        self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.fig.canvas.mpl_connect('key_press_event', self.keyentry)

        self.diagnostic=0
        self.vmin=np.nanmin(self.diagnostics[0])-0.05*np.nanmin(self.diagnostics[0])
        self.vmax=np.nanmax(self.diagnostics[0])+0.05*np.nanmax(self.diagnostics[0])
        # plot the diagnostics
        self.cmap=get_cmap(self)
        self.map=plot_map(self, self.diagnostics[0])

        # setup spectrum plot window
        self.plot_spectrum,=plot_spectrum(self,[0,1], [0,0],label='spec',ls='-',drawstyle='steps')
        # setup spectrum plot window
        self.plot_res,=plot_spectrum(self,[0,1], [0,0],label='residual',ls='-',drawstyle='steps',color='orange')
        # plot the spectrum and the smoothed spectrum
        self.plot_smooth,=plot_spectrum(self,[0,1],[0,0],label='smoothed spec',lw=1.5,ls=':',color='k')
        # plot the signal to noise threshold
        self.plot_SNR,=plot_snr(self,[0,1],[0,0],linestyle='--',color='k',label='SNR*rms')
        # plot the predicted peak locations
        self.plot_peak_markers,=plot_peak_locations(self,[0],[0],color='k',linestyle='',marker='o',markersize=5,label='dspec prediction')
        # plot stems for the markers
        self.plot_peak_lines=plot_stems(self,[0],[0],color='k',ls='-')
        # plot model
        self.plot_tot,=plot_model(self,[0,1],[0,0],color='magenta',ls='-',lw=1,label='total model',)
        self.plot_model=[]

        # add some text
        self.text_model_info=print_information(self,0.5,-0.2,'Select a spectrum', fontsize=10)

        #=======#
        # sliders
        #=======#
        # create sliders for controlling the imshow limits
        self.slider_vmin_ax=self.fig.add_axes([0.05, 0.84, 0.275, 0.015])
        self.slider_vmin=make_slider(self.slider_vmin_ax,"vmin",self.vmin,self.vmax,self.update_vmin,valinit=self.vmin,valfmt="%1.2f", facecolor='0.75')
        self.slider_vmax_ax=self.fig.add_axes([0.05, 0.8125, 0.275, 0.015])
        self.slider_vmax=make_slider(self.slider_vmax_ax,"vmax",self.vmin,self.vmax,self.update_vmax,valinit=self.vmax,valfmt="%1.2f", facecolor='0.75')

        self.slider_snr_ax=self.fig.add_axes([0.675, 0.84, 0.275, 0.015])
        self.slider_snr=make_slider(self.slider_snr_ax,"SNR",self.minSNR,self.maxSNR,self.update_SNR,valinit=self.SNR, valfmt="%i", facecolor='0.75')

        self.slider_alpha_ax=self.fig.add_axes([0.675, 0.8125, 0.275, 0.015])
        self.slider_alpha=make_slider(self.slider_alpha_ax,"alpha",self.minalpha,self.maxalpha,self.update_alpha,valinit=self.alpha, valfmt="%i", facecolor='0.75')

        #========================#
        # compute diagnostics menu
        #========================#
        textboxheight=0.025
        textboxwidth=0.05
        top=0.78
        mid=0.06
        space=0.025
        smallspace=space/2.

        #====================#
        # plot moments menu
        #====================#
        space=0.0128
        size=0.035

        # Controls which moment map gets plotted
        start=0.05
        self.diag0_ax=self.fig.add_axes([start, 0.15, size, size])
        self.diag0=make_button(self.diag0_ax,"rms",lambda event: self.update_map(event, map=0),color='0.75',hovercolor='0.95')
        end=start+size+space

        start=end
        self.diag1_ax=self.fig.add_axes([start, 0.15, size, size])
        self.diag1=make_button(self.diag1_ax,"residstd",lambda event: self.update_map(event, map=1),color='0.75',hovercolor='0.95')
        end=start+size+space

        start=end
        self.diag2_ax=self.fig.add_axes([start, 0.15, size, size])
        self.diag2=make_button(self.diag2_ax,"redchisq",lambda event: self.update_map(event, map=2),color='0.75',hovercolor='0.95')
        end=start+size+space

        start=end
        self.diag3_ax=self.fig.add_axes([start, 0.15, size, size])
        self.diag3=make_button(self.diag3_ax,"ncomps",lambda event: self.update_map(event, map=3),color='0.75',hovercolor='0.95')
        end=start+size+space

        start=end
        self.diag4_ax=self.fig.add_axes([start, 0.15, size, size])
        self.diag4=make_button(self.diag4_ax,"aic",lambda event: self.update_map(event, map=4),color='0.75',hovercolor='0.95')
        end=start+size+space

        start=end
        self.diag5_ax=self.fig.add_axes([start, 0.15, size, size])
        self.diag5=make_button(self.diag5_ax,"chisq",lambda event: self.update_map(event, map=5),color='0.75',hovercolor='0.95')
        end=start+size+space

        #====================
        # model selection
        #====================
        self.modelindex=0
        self.models=[]
        inc=0.0975
        self.button_skipbackone=self.fig.add_axes([0.675+inc, 0.3875, 0.025, 0.025])
        self.button_skipbackone=make_button(self.button_skipbackone,"<",lambda event: self.new_model(event, _type='previous'))
        #
        self.textbox_index=self.fig.add_axes([0.7025+inc, 0.3875, 0.025, 0.025])
        self.textbox_index=make_textbox(self.textbox_index,'',str(self.modelindex+1),self.change_text)
        #
        self.button_skipforwardone=self.fig.add_axes([0.73+inc, 0.3875, 0.025, 0.025])
        self.button_skipforwardone=make_button(self.button_skipforwardone,">",lambda event: self.new_model(event, _type='next'))


        self.button_fit_dspec_ax=self.fig.add_axes([0.7325, 0.315, 0.05, 0.05])
        self.button_fit_dspec=make_button(self.button_fit_dspec_ax,"fit (dspec)",self.dspec_manual)

        self.button_fit_manual_ax=self.fig.add_axes([0.7875, 0.315, 0.05, 0.05])
        self.button_fit_manual=make_button(self.button_fit_manual_ax,"fit (manual)",self.open_scousefitter_manual)

        self.button_remove_fit_ax=self.fig.add_axes([0.8425, 0.315, 0.05, 0.05])
        self.button_remove_fit=make_button(self.button_remove_fit_ax,"remove fit",self.remove_fit)

        self.confirm_ax=self.fig.add_axes([0.7875, 0.2425, 0.05, 0.05])
        self.confirm_button=make_button(self.confirm_ax,"confirm",self.confirm_model, color='palegreen',hovercolor='springgreen')

        #================
        # Continue button
        #================
        self.continue_ax=self.fig.add_axes([0.9, 0.14, 0.05, 0.05])
        self.continue_button=make_button(self.continue_ax,"continue",self.check_complete, color='lightblue',hovercolor='aliceblue')

    def show(self):
        """
        Show the plot
        """
        import matplotlib.pyplot as plt
        plt.show()

    def close_window(self):
        """
        Closes the plot window
        """
        import matplotlib.pyplot as plt
        if 'q' not in plt.rcParams['keymap.quit']:
            plt.rcParams['keymap.quit'].append('q')
        if 'Q' not in plt.rcParams['keymap.quit_all']:
            plt.rcParams['keymap.quit_all'].append('Q')
        plt.close('all')

    def check_complete(self, event):
        """
        Controls what happens if continue button is pressed.

        Parameters
        ----------
        event : button press event

        """
        if np.size(self.check_spec_indices)==0:
            if self.verbose:
                print(colors.fg._lightgreen_+"No spectra were selected for modification. Original model solutions retained.   "+colors._endc_)
                print('')
        else:
            if self.verbose:
                print(colors.fg._lightgreen_+"Procedure complete. {} spectra were selected for close inspection".format(np.size(self.check_spec_indices))+colors._endc_)
                print('')
        # close the window
        self.close_window()

    def click(self, event):
        """
        Controls what happens when the interactive plot window is clicked
        """
        # create a list containing all axes
        axislist=[self.map_window]+self.spec_grid_window

        # retrieve the index of the axis that has been clicked
        i = 0
        axisNr = None
        for axis in axislist:
            if axis == event.inaxes:
                axisNr=i
                break
            i+=1

        # make sure the click is registered
        eventdata=np.asarray([event.xdata, event.ydata])
        if not None in eventdata:
            # identify the location of the click
            self.xpos,self.ypos=int(event.xdata+0.5), int(event.ydata+0.5)

            # What happens depends on which axis has been selected
            if axisNr == 0:
                # left mouse click
                if event.button == 1:
                    # get the flattened indices of the pixel and its neighbours
                    self.keys=get_neighbours(self)

                    # if the map is selected then we are going to plot
                    # some spectra
                    plot_spectra(self, self.scouseobject, color='limegreen')

            # else if one of the spectra are selected then store the
            # information and identify the spectrum as one to be
            # checked during stage 6
            elif axisNr in np.arange(1,np.size(axislist)):
                self.select_spectra(event, axisNr)

            else:
                pass
            self.fig.canvas.draw()

    def keyentry(self, event):
        # create a list containing all axes
        axislist=[self.map_window]+self.spec_grid_window

        # retrieve the index of the axis that has been clicked
        i = 0
        axisNr = None
        for axis in axislist:
            if axis == event.inaxes:
                axisNr=i
                break
            i+=1

        # make sure the key entry is registered
        eventdata=np.asarray([event.xdata, event.ydata])
        if not None in eventdata:
            # identify the location of the click
            self.xpos,self.ypos=int(event.xdata+0.5), int(event.ydata+0.5)
            # What happens depends on which axis has been selected
            if axisNr == 0:
                # enter key
                if event.key == 'enter':
                    # get the flattened indices of the pixel and its neighbours
                    self.keys=get_neighbours(self)
                    # if the map is selected then we are going to plot
                    # some spectra
                    plot_spectra(self, self.scouseobject, color='limegreen')

            # else if one of the spectra are selected then store the
            # information and identify the spectrum as one to be
            # checked during stage 6
            elif axisNr in np.arange(1,np.size(axislist)):
                self.select_spectra(event, axisNr)

            else:
                pass

            self.fig.canvas.draw()

    def update_vmin(self,pos=None):
        """
        Controls what happens when the vmin slider is updated

        Parameters
        ----------
        pos : slider position
        """
        import matplotlib.pyplot as plt
        # set the upper limits otherwise it'll go a bit weird
        if pos > self.vmax:
            self.vmin=self.vmax
        else:
            self.vmin = pos

        maptoplot=self.diagnostics[self.diagnostic]

        # plot the map with the new slider values
        self.cmap=get_cmap(self)
        self.map=plot_map(self,maptoplot,update=True)
        # update plot
        self.fig.canvas.draw()

    def update_vmax(self,pos=None):
        """
        Controls what happens when the vmax slider is updated

        Parameters
        ----------
        pos : slider position
        """
        import matplotlib.pyplot as plt
        # set the upper limits otherwise it'll go a bit weird
        if pos < self.vmin:
            self.vmax=self.vmin
        else:
            self.vmax=pos

        maptoplot=self.diagnostics[self.diagnostic]

        # plot the map with the new slider values
        self.cmap=get_cmap(self)
        self.map=plot_map(self,maptoplot,update=True)
        # update plot
        self.fig.canvas.draw()

    def update_map(self,event,map=None):
        """
        Controls what happens when one of the map buttons is pressed

        Parameters
        ----------
        event : button press event
        map : number
            Index for the self.moments list - indicates which map to plot
        """
        import matplotlib.pyplot as plt
        # Get the map
        self.diagnostic=map
        maptoplot=self.diagnostics[self.diagnostic]

        # update the Limits
        if self.diagnostic==3:
            self.vmin = np.nanmin(maptoplot)
            if self.vmin==0.0:
                self.vmin=1.0
            self.vmax = np.nanmax(maptoplot)
        else:
            self.vmin = np.nanmin(maptoplot)-0.05*np.nanmin(maptoplot)
            self.vmax = np.nanmax(maptoplot)+0.05*np.nanmax(maptoplot)
        # update the sliders
        update_sliders(self)
        # plot the map
        self.cmap=get_cmap(self)
        self.map=plot_map(self,maptoplot,update=True)
        # update plot
        self.fig.canvas.draw()

    def select_spectra(self, event, axisNr):
        """
        Controls what happens when a spectrum is selected
        """

        if self.keys is not None:
            self.speckey=self.keys[axisNr-1]
            ax=self.spec_grid_window[axisNr-1]

            if event.name=='button_press_event':

                # select
                if event.button==1:
                    if ~np.isnan(self.speckey):
                        ax.patch.set_facecolor('red')
                        ax.patch.set_alpha(0.1)
                        self.get_spectral_info()
                        self.spectrum_selected(self.speckey)
                        if self.speckey not in self.check_spec_indices:
                            self.check_spec_indices.append(self.speckey)



                # # deselect
                # elif event.button==3:
                #     if ~np.isnan(key):
                #          ax.patch.set_facecolor('white')
                #          ax.patch.set_alpha(0.0)
                #          if key in self.check_spec_indices:
                #              self.check_spec_indices.remove(key)

                else:
                    return

            elif event.name=='key_press_event':

                # select
                if (event.key=='enter'):
                    if ~np.isnan(self.speckey):
                        ax.patch.set_facecolor('red')
                        ax.patch.set_alpha(0.1)
                        self.get_spectral_info()
                        self.spectrum_selected(self.speckey)
                        if self.speckey not in self.check_spec_indices:
                            self.check_spec_indices.append(self.speckey)


                # deselect
                # elif (event.key=='d') or (event.key=='r'):
                #     if ~np.isnan(key):
                #          ax.patch.set_facecolor('white')
                #          ax.patch.set_alpha(0.0)
                #          if key in self.check_spec_indices:
                #              self.check_spec_indices.remove(key)

                # select all
                # elif (event.key=='a'):
                #     for axid, _key in enumerate(self.keys):
                #         if ~np.isnan(_key):
                #             self.spec_grid_window[axid].patch.set_facecolor('red')
                #             self.spec_grid_window[axid].patch.set_alpha(0.1)
                #             if _key not in self.check_spec_indices:
                #                 self.check_spec_indices.append(_key)

                # remove all
                # elif (event.key=='backspace') or (event.key=='escape'):
                #     for axid, _key in enumerate(self.keys):
                #         if ~np.isnan(_key):
                #             self.spec_grid_window[axid].patch.set_facecolor('white')
                #             self.spec_grid_window[axid].patch.set_alpha(0.0)
                #             if _key in self.check_spec_indices:
                #                 self.check_spec_indices.remove(_key)

                else:
                    return

    def get_spectral_info(self):
        from .model_housing import individual_spectrum

        if self.speckey not in self.scouseobject.indiv_dict.keys():
            index=np.unravel_index(self.speckey,self.scouseobject.cube.shape[1:])
            self.my_spectrum=individual_spectrum(np.array([index[1],index[0]]),self.scouseobject.cube.filled_data[:,index[0],index[1]].value,index=self.speckey,
                                scouseobject=self.scouseobject, saa_dict_index=None,
                                saaindex=None)
            self.scouseobject.indiv_dict[self.speckey]=self.my_spectrum
            setattr(self.my_spectrum,'model_from_parent',[None],)
        else:
            self.my_spectrum=self.scouseobject.indiv_dict[self.speckey]

        self.specx=self.scouseobject.xtrim
        self.specy=self.my_spectrum.spectrum[self.scouseobject.trimids]
        self.specrms=self.my_spectrum.rms
        self.dsp = compute_dsp(self)
        self.modelindex=0
        self.models=get_model_list(self)

    def spectrum_selected(self, key):
        """
        Controls what happens when a spectrum has been selected
        """

        update_text(self.text_model_info,'')
        # update plots
        ymin=np.nanmin(self.specy)-0.2*np.nanmax(self.specy)
        ymax=np.nanmax(self.specy)+0.2*np.nanmax(self.specy)
        self.spectrum_window.set_ylim([ymin,ymax])
        plot_spectrum(self,self.specx, self.specy,plottoupdate=self.plot_spectrum,update=True,ls='-')
        plot_spectrum(self,self.specx,self.ysmooth,update=True,plottoupdate=self.plot_smooth,ls=':')
        plot_snr(self,[np.nanmin(self.specx),np.nanmax(self.specx)],[self.SNR*self.specrms, self.SNR*self.specrms],update=True,plottoupdate=self.plot_SNR)
        plot_peak_locations(self,self.centroids,self.peaks,update=True,plottoupdate=self.plot_peak_markers,markersize=5)
        plot_stems(self,self.centroids,self.peaks,update=True, color='k',ls='-')
        self.textbox_index.set_val(str(self.modelindex+1))
        update_text(self.text_model_info,'There are '+str(int(len(self.models)))+' models available for this spectrum')

        # recreate the model
        if self.my_spectrum.model is not None:
            self.mod, self.res, self.totmod=recreate_model(self,self.my_spectrum,self.my_spectrum.model)
        else:
            self.mod, self.res, self.totmod=np.zeros([len(self.specx), int(1)]),self.specy,np.zeros([len(self.specx), int(1)])

        plot_model(self, self.specx, self.totmod, update=True, plottoupdate=self.plot_tot)
        plot_model(self,self.specx, self.res,plottoupdate=self.plot_res,update=True,ls='-')

        if np.size(self.plot_model)!=0:
            for i in range(len(self.plot_model)):
                self.plot_model[i].pop(0).remove()

        if self.my_spectrum.model is not None:
            self.plot_model=[]
            for k in range(np.shape(self.mod)[1]):
                # plot individual components
                self.plot_model_indiv=plot_model(self, self.specx,self.mod[:,k],color='limegreen',ls='-',lw=1,label='comp '+str(k))
                self.plot_model.append(self.plot_model_indiv)

        # plot a legend
        self.spectrum_legend = self.spectrum_window.legend(loc=2,frameon=False,fontsize=8)
        self.spectrum_window_lookup_artist, self.spectrum_window_lookup_handle = self.build_legend_lookups(self.spectrum_legend)
        self.setup_legend_connections(self.spectrum_legend, self.spectrum_window_lookup_artist, self.spectrum_window_lookup_handle)
        self.update_legend(self.spectrum_window_lookup_artist, self.spectrum_window_lookup_handle)

        self.fig.canvas.draw()

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
        plot_snr(self,[np.nanmin(self.specx),np.nanmax(self.specx)],[self.SNR*self.specrms, self.SNR*self.specrms],update=True,plottoupdate=self.plot_SNR)
        plot_peak_locations(self,self.centroids,self.peaks,update=True,plottoupdate=self.plot_peak_markers,markersize=5)
        plot_stems(self,self.centroids,self.peaks,update=True, color='k', ls='-')

        # update plot
        self.fig.canvas.draw()

    def update_alpha(self,pos=None):
        """
        This controls what happens if the alpha slider is updated

        Parameters:
        -----------
        pos : position on the slider

        """
        # new kernel size
        self.alpha=int(round(pos))
        #compute new dsp
        self.dsp = compute_dsp(self)
        # update spectrum plot
        plot_spectrum(self,self.specx,self.ysmooth,update=True,plottoupdate=self.plot_smooth,ls=':')
        plot_peak_locations(self,self.centroids,self.peaks,update=True,plottoupdate=self.plot_peak_markers,markersize=5)
        plot_stems(self,self.centroids,self.peaks,update=True, color='k',ls='-')

        # update plot
        self.fig.canvas.draw()

    def change_text(self, text):
        self.modeldict=None
        # extract value from text input.
        value = eval(text)
        # correct the indexing from base 1 to base 0 (with the former making
        # more sense for the navigator)
        value = value-1
        # Control what happens if you are at the start or end
        if value < 0:
            value=0
        if value > len(self.models)-1:
            value=len(self.models)-1
        # set the index to the new one selected by the navigator
        self.modelindex=value
        # recreate the model
        self.mod, self.res, self.totmod=recreate_model(self,self.my_spectrum,self.models[self.modelindex])
        plot_model(self, self.specx, self.totmod, update=True, plottoupdate=self.plot_tot)
        plot_model(self, self.specx, self.res, update=True, plottoupdate=self.plot_res)
        if np.size(self.plot_model)!=0:
            for i in range(len(self.plot_model)):
                self.plot_model[i].pop(0).remove()

        self.plot_model=[]
        for k in range(np.shape(self.mod)[1]):
            # plot individual components
            self.plot_model_indiv=plot_model(self, self.specx,self.mod[:,k],color='limegreen',ls='-',lw=1,label='comp '+str(k))
            self.plot_model.append(self.plot_model_indiv)

        # update plot
        self.fig.canvas.draw()

    def new_model(self,event,_type=None):
        """
        This descibes what happens when a new model is selected using the
        buttons "previous" and "next"

        Parameters
        ----------
        event : button click event
        _type : describes what button has been pressed

        """
        value=update_index(self,_type)
        value += 1
        self.textbox_index.set_val(str(value))

    def setup_legend_connections(self, legend, lookup_artist,lookup_handle):
        """
        setting up the connections for the interactive plot legend

        Parameters
        ----------
        legend : matplotlib legend
        lookup_artist : matplotlib artist
        lookup_handle : matplotlib legend handles

        """
        for artist in legend.legendHandles:
            artist.set_picker(True)
            artist.set_pickradius(10) # 10 points tolerance

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

    def open_scousefitter_manual(self, event):
        """
        This controls the manual fitting if no reasonable solution can be found
        with the derivative spectroscopy method

        """
        initiate_decomposer(self)
        # manual fit
        Decomposer.fit_spectrum_manually(self.decomposer, fittype=self.fittype)
        # add model to dictionary
        self.modeldict=get_model_info(self)
        # recreate the model
        self.mod,self.res,self.totmod=recreate_model_manual(self)
        # update the plot with the manually-fitted solution
        plot_model(self, self.specx, self.totmod, update=True, plottoupdate=self.plot_tot)
        plot_model(self,self.specx, self.res,plottoupdate=self.plot_res,update=True,ls='-')

        if np.size(self.plot_model)!=0:
            for i in range(len(self.plot_model)):
                self.plot_model[i].pop(0).remove()

        self.plot_model=[]
        for k in range(np.shape(self.mod)[1]):
            # plot individual components
            self.plot_model_indiv=plot_model(self, self.specx,self.mod[:,k],color='limegreen',ls='-',lw=1,label='comp '+str(k))
            self.plot_model.append(self.plot_model_indiv)

        # plot a legend
        self.spectrum_legend.remove()
        self.spectrum_legend = self.spectrum_window.legend(loc=2,frameon=False,fontsize=8)
        self.spectrum_window_lookup_artist, self.spectrum_window_lookup_handle = self.build_legend_lookups(self.spectrum_legend)
        self.setup_legend_connections(self.spectrum_legend, self.spectrum_window_lookup_artist, self.spectrum_window_lookup_handle)
        self.update_legend(self.spectrum_window_lookup_artist, self.spectrum_window_lookup_handle)

        # update plot
        self.fig.canvas.draw()

    def dspec_manual(self, event):
        """
        This controls the manual dspec fitter.

        """
        initiate_decomposer(self)
        # Fit the spectrum according to dspec guesses
        if self.dsp.ncomps != 0:
            Decomposer.fit_spectrum_with_guesses(self.decomposer,self.guesses,fittype=self.fittype)
        # Add the best-fitting solution and useful parameters to a dictionary
        self.modeldict=get_model_info(self)
        # recreate the model
        self.mod,self.res,self.totmod=recreate_model_manual(self)
        # update the plot with the manually-fitted solution
        plot_model(self, self.specx, self.totmod, update=True, plottoupdate=self.plot_tot)
        plot_model(self,self.specx, self.res,plottoupdate=self.plot_res,update=True,ls='-')

        if np.size(self.plot_model)!=0:
            for i in range(len(self.plot_model)):
                self.plot_model[i].pop(0).remove()

        self.plot_model=[]
        for k in range(np.shape(self.mod)[1]):
            # plot individual components
            self.plot_model_indiv=plot_model(self, self.specx,self.mod[:,k],color='limegreen',ls='-',lw=1,label='comp '+str(k))
            self.plot_model.append(self.plot_model_indiv)

        # plot a legend
        self.spectrum_legend.remove()
        self.spectrum_legend = self.spectrum_window.legend(loc=2,frameon=False,fontsize=8)
        self.spectrum_window_lookup_artist, self.spectrum_window_lookup_handle = self.build_legend_lookups(self.spectrum_legend)
        self.setup_legend_connections(self.spectrum_legend, self.spectrum_window_lookup_artist, self.spectrum_window_lookup_handle)
        self.update_legend(self.spectrum_window_lookup_artist, self.spectrum_window_lookup_handle)

        # update plot
        self.fig.canvas.draw()

    def remove_fit(self, event):
        """
        This controls the removal of a model

        """
        initiate_decomposer(self)
        # Add the best-fitting solution and useful parameters to a dictionary
        self.modeldict=get_model_info(self)
        self.modeldict['method']='remove'
        # recreate the model
        self.mod,self.res,self.totmod=recreate_model_manual(self)
        # update the plot with the manually-fitted solution
        plot_model(self, self.specx, self.totmod, update=True, plottoupdate=self.plot_tot)
        plot_model(self,self.specx, self.res,plottoupdate=self.plot_res,update=True,ls='-')

        if np.size(self.plot_model)!=0:
            for i in range(len(self.plot_model)):
                self.plot_model[i].pop(0).remove()

        # plot a legend
        self.spectrum_legend.remove()
        self.spectrum_legend = self.spectrum_window.legend(loc=2,frameon=False,fontsize=8)
        self.spectrum_window_lookup_artist, self.spectrum_window_lookup_handle = self.build_legend_lookups(self.spectrum_legend)
        self.setup_legend_connections(self.spectrum_legend, self.spectrum_window_lookup_artist, self.spectrum_window_lookup_handle)
        self.update_legend(self.spectrum_window_lookup_artist, self.spectrum_window_lookup_handle)

        # update plot
        self.fig.canvas.draw()

    def confirm_model(self, event):
        """
        Controls what happens when the confirm button is pressed
        """
        from .model_housing import indivmodel

        if self.modeldict is None:
            if self.my_spectrum.model is None:
                return
            else:
                model=self.models[self.modelindex]
        else:
            model=indivmodel(self.modeldict)

        setattr(self.my_spectrum,'model',model,)
        setattr(self.my_spectrum,'decision',model.method,)

        self.models=get_model_list(self)
        update_text(self.text_model_info,'There are '+str(int(len(self.models)))+' models available for this spectrum')
        self.modelindex=0
        self.textbox_index.set_val(str(self.modelindex+1))
        self.fig.canvas.draw()

        if model.ncomps==0:
            for i in range(np.shape(self.diagnostics)[0]):
                self.diagnostics[i][self.my_spectrum.coordinates[1],self.my_spectrum.coordinates[0]]=np.nan
        else:
            self.diagnostics[0][self.my_spectrum.coordinates[1],self.my_spectrum.coordinates[0]]=model.rms
            self.diagnostics[1][self.my_spectrum.coordinates[1],self.my_spectrum.coordinates[0]]=model.residstd
            self.diagnostics[2][self.my_spectrum.coordinates[1],self.my_spectrum.coordinates[0]]=model.redchisq
            self.diagnostics[3][self.my_spectrum.coordinates[1],self.my_spectrum.coordinates[0]]=model.ncomps
            self.diagnostics[4][self.my_spectrum.coordinates[1],self.my_spectrum.coordinates[0]]=model.AIC
            self.diagnostics[5][self.my_spectrum.coordinates[1],self.my_spectrum.coordinates[0]]=model.chisq

        plot_spectra(self, self.scouseobject, color='limegreen')

        self.update_map(None, map=self.diagnostic)
        save_maps(self,self.diagnostics)

        self.modeldict=None

def setup_plot_window(self,ax,color='white'):
    """
    GUI setup
    """

    window=self.fig.add_axes(ax)
    window.tick_params( axis='both',          # changes apply to the both axes
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        left=False,
                        right=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        labelbottom=False )

    window.set_facecolor(color)
    window.text(0.5,1.2,'Select region', ha='center')
    window.text(0.5,1.15,"To select: Left click or press 'enter'", ha='center')
    return window

def setup_map_window(self):
    """
    Setup the map window for plotting
    """
    try:
        from astropy.visualization.wcsaxes import WCSAxes
        self._wcaxes_imported = True
    except ImportError:
        self._wcaxes_imported = False
        if self.moments[0].wcs is not None:
            warnings.warn("`WCSAxes` required for wcs coordinate display.")

    newaxis=[self.blank_window_ax[0]+0.03, self.blank_window_ax[1]+0.03, self.blank_window_ax[2]-0.06,self.blank_window_ax[3]-0.045]

    if self.scouseobject.cube.wcs is not None and self._wcaxes_imported:
        wcs=self.scouseobject.cube.wcs
        wcs=wcs.dropaxis(2)

        ax_image = WCSAxes(self.fig, newaxis, wcs=wcs, slices=('x','y'))
        map_window = self.fig.add_axes(ax_image)
        x = map_window.coords[0]
        y = map_window.coords[1]
        x.set_axislabel('  ')
        y.set_axislabel('  ')
        x.set_ticklabel(exclude_overlapping=True)
        y.set_ticklabel(rotation=90,verticalalignment='bottom', horizontalalignment='left',exclude_overlapping=True)
    else:
        map_window = self.fig.add_axes(newaxis)
    return map_window

def compute_diagnostic_plots(self,):
    """
    Create diagnostic maps

    """
    from astropy import units as u
    from .verbose_output import print_to_terminal

    # we output the maps as fits files so first check if they exist. If they
    # do then load them rather than create them twice
    if os.path.exists(self.scouseobject.outputdirectory+self.scouseobject.filename+'/stage_4/stage_4_'+self.maps[0]+'.fits'):
        if self.verbose:
            progress_bar = print_to_terminal(stage='s4', step='diagnosticsload')
        diagnostics=load_maps(self)
    else:
        if self.verbose:
            progress_bar = print_to_terminal(stage='s4', step='diagnosticsinit')
        diagnostics=[generate_2d_parametermap(self, mapname) for mapname in self.maps]
        if self.verbose:
            print("")

    return diagnostics

def generate_2d_parametermap(self, spectrum_parameter):
    """
    Create a 2D map of a given spectral parameter

    Parameters
    ----------
    scouseobject : Instance of the scousepy class

    """
    from .verbose_output import print_to_terminal
    map = np.zeros(self.scouseobject.cube.shape[1:])
    map[:] = np.nan

    if self.verbose:
        progress_bar = print_to_terminal(stage='s4', step='diagnostics',length=len(self.scouseobject.indiv_dict.keys()))

    for key,spectrum in self.scouseobject.indiv_dict.items():
        cx,cy = spectrum.coordinates
        #if getattr(spectrum.model, 'ncomps') != 0:
        map[cy, cx] = getattr(spectrum.model, spectrum_parameter)
        #else:
        #    map[cy, cx] = np.nan
        if self.verbose:
            progress_bar.update()

    return map

def save_maps(self, diagnostics, overwrite=True):
    """
    Procedure to save the output maps as fits files

    Parameters
    ----------
    diagnostics : list
        a list containing the diagnostic plots
    overwrite : bool
        optional overwrite

    """
    from astropy.io import fits
    savedir=self.scouseobject.outputdirectory+self.scouseobject.filename+'/stage_4/'
    for index,mapname in enumerate(self.maps):
        fh = fits.PrimaryHDU(data=diagnostics[index], header=self.scouseobject.cube[0,:,:].header)
        fh.writeto(os.path.join(savedir, "stage_4_"+mapname+".fits"), overwrite=overwrite)

def load_maps(self):
    """
    Procedure to load the maps

    """
    from astropy.io import fits
    savedir=self.scouseobject.outputdirectory+self.scouseobject.filename+'/stage_4/'

    return [fits.getdata(savedir+'stage_4_'+mapname+'.fits') for mapname in self.maps]

def get_cmap(self):
    import matplotlib as mpl
    import copy
    import matplotlib.pyplot as plt
    if self.diagnostic==3:
        self.cmap=copy.copy(mpl.cm.get_cmap("viridis"))
        self.cmap.set_bad(color='lightgrey')
        self.cmap.set_under('w')
    else:
        self.cmap=copy.copy(mpl.cm.get_cmap("viridis"))
        self.cmap.set_bad(color='lightgrey')

def plot_map(self, map, update=False):
    """
    map plotting

    Parameters
    ----------
    map : ndarray
        map to plot
    update : Bool
        updating the map or plotting from scratch

    """
    import matplotlib.pyplot as plt
    if update:
        empty=np.empty(np.shape(self.diagnostics[self.diagnostic]))
        empty[:]=np.nan
        self.map.set_data(empty)
        return self.map_window.imshow(map, origin='lower', interpolation='nearest',cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
    else:
        return self.map_window.imshow(map, origin='lower', interpolation='nearest',cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)

def get_neighbours(self):
    """
    Returns a list of flattened indices for a given spectrum and its neighbours

    Parameters
    ----------
    xpos : number
        x position of the selected pixel
    ypos : number
        y position of the selected pixel

    """
    shape=self.scouseobject.cube.shape[1:]
    neighboursx=np.arange(self.xpos-(self.blocksize-1)/2,(self.xpos+(self.blocksize-1)/2)+1,dtype='int' )
    neighboursx=[x if (x>=0) & (x<=shape[1]-1) else np.nan for x in neighboursx ]
    neighboursy=np.arange(self.ypos-(self.blocksize-1)/2,(self.ypos+(self.blocksize-1)/2)+1,dtype='int' )
    neighboursy=[y if (y>=0) & (y<=shape[0]-1) else np.nan for y in neighboursy ]
    keys=[np.ravel_multi_index([y,x], shape)  if np.all(np.isfinite(np.asarray([y,x]))) else np.nan for y in neighboursy for x in neighboursx]

    return keys

def setup_spec_window(self):
    width=self.spec_window_ax[2]
    height=self.spec_window_ax[3]

    bottomleftx=self.spec_window_ax[0]
    bottomlefty=self.spec_window_ax[1]

    specwidth=width/self.blocksize
    specheight=height/self.blocksize - 0.0084

    axlist=[]
    for i in range(int(self.blocksize)):
        for j in range(int(self.blocksize)):
            axloc=[bottomleftx+(j*specwidth)+(j*0.00575), bottomlefty+(i*specheight)+(i*0.01), specwidth, specheight]
            axis=self.fig.add_axes(axloc)
            axis.tick_params(axis='both',          # changes apply to the both axes
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            left=False,
                            right=False,
                            labelleft=False,
                            labelright=False,
                            labeltop=False,
                            labelbottom=False )
            axlist.append(axis)

    self.blank_window.text(1.625,1.20,'Select spectra', ha='center')
    self.blank_window.text(1.625,1.15,"To select single: Left click or press 'enter'", ha='center')
    #self.blank_window.text(1.6,1.10,"To de-select: Right click or press 'r' or 'd'", ha='center')
    #self.blank_window.text(1.625,1.10,"To select all: press 'a'", ha='center')
    return axlist

def plot_spectra(self,scouseobject, color='green'):
    """
    Plotting spectra once the map has been clicked

    Parameters
    ----------
    keys : list
        A list of indices refering to the pixel locations in the map
    """
    # loop through the spectra
    for i in range(self.blocksize):
        for j in range(self.blocksize):
            key=self.keys[j+i*self.blocksize]
            ax=self.spec_grid_window[j+i*self.blocksize]

            ax.patch.set_facecolor('lightgrey')
            ax.patch.set_alpha(1.0)

            # remove anything that is currently plotted
            lines=ax.get_lines()
            for line in lines:
                line.set_xdata([])
                line.set_ydata([])

            # Key values that are outside the map limits are set to nan
            if ~np.isnan(key):
                if (key in self.check_spec_indices) or (key in self.selected_spectra):
                    ax.patch.set_facecolor('red')
                    ax.patch.set_alpha(0.1)
                else:
                    ax.patch.set_facecolor('white')
                    ax.patch.set_alpha(1.0)
                # get the 2D index
                index=np.unravel_index(key,self.scouseobject.cube.shape[1:])
                # plot from the cube rather than the fitted spectrum
                spectrum=self.scouseobject.cube.filled_data[scouseobject.trimids,index[0],index[1]].value
                # redefine the axis limits and plot the spectrum
                ax.set_xlim(np.min(self.scouseobject.xtrim), np.max(self.scouseobject.xtrim))
                ax.set_ylim(np.nanmin(spectrum), 1.05*np.nanmax(spectrum))
                ax.plot(scouseobject.xtrim,spectrum, drawstyle='steps', color='k', lw=0.85)

                # now check to see if a model is available
                if key in scouseobject.indiv_dict.keys():
                    indivspec=scouseobject.indiv_dict[key]
                    # recreate the model
                    if indivspec.model is not None:
                        mod, res, totmod=recreate_model(self,indivspec,indivspec.model)
                        ax.plot(scouseobject.xtrim,res, color='orange', lw=0.5, drawstyle='steps')
                        for k in range(np.shape(mod)[1]):
                            # plot individual components
                            ax.plot(self.scouseobject.xtrim,mod[:,k], color=color, lw=1)

    self.fig.canvas.draw()

def update_index(self,_type):
    """
    Updates the index for the navigator
    """
    if _type=='next':
        if self.modelindex==len(self.models)-1:
            value = len(self.models)-1
        else:
            value = self.modelindex+1
    elif _type=='previous':
        if self.modelindex==0:
            value = 0
        else:
            value = self.modelindex-1
    else:
        pass
    return value

def compute_dsp(self):
    """
    Computes derivative spectroscopy and sets some global values
    """
    from scousepy.dspec import DSpec
    dsp = DSpec(self.specx,self.specy,self.specrms,SNR=self.SNR,alpha=self.alpha)
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
    modeldict['alpha']=self.alpha

    return modeldict

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

def initiate_decomposer(self):
    """
    initiates an instance of the SpectralDecomposer
    """
    # create the decomposer
    self.decomposer=Decomposer(self.specx, self.specy, self.specrms)
    Decomposer.create_a_spectrum(self.decomposer,unit=self.unit,xarrkwargs=self.xarrkwargs)
    # generate pyspeckit spectrum
    self.spectrum=self.decomposer.pskspectrum

def get_model_list(self):
    model=self.my_spectrum.model
    if model is None:
        modellist=[]
        return modellist

    modelaic=model.AIC
    modellist=[_model for _model in self.my_spectrum.model_from_parent if _model is not None]
    aiclist=[_model.AIC for _model in modellist]

    if modelaic in aiclist:
        idaic=np.squeeze(np.where(aiclist==modelaic))
        del modellist[idaic]
        del aiclist[idaic]

    modelarr=np.asarray(modellist)[::-1]
    aicarr=np.asarray(aiclist)[::-1]

    if np.size(modellist)!=0:
        sortedids=np.argsort(aicarr)
        modelarr = modelarr[sortedids]
        modellist=list(modelarr)

    modellist.insert(0,model)

    return modellist

def recreate_model(self, indivspec, model):
    """
    Recreates model from parameters in modeldict

    """
    import pyspeckit
    from scousepy.SpectralDecomposer import Decomposer

    if model.ncomps != 0.0:
        decomposer=Decomposer(self.scouseobject.xtrim, indivspec.spectrum[self.scouseobject.trimids], indivspec.rms)
        Decomposer.create_a_spectrum(decomposer)
        pskspectrum=decomposer.pskspectrum
        pskspectrum.specfit.fittype=model.fittype
        pskspectrum.specfit.fitter = pskspectrum.specfit.Registry.multifitters[model.fittype]
        mod = np.zeros([len(self.scouseobject.xtrim), int(model.ncomps)])
        for k in range(int(model.ncomps)):
            modparams = model.params[(k*len(model.parnames)):(k*len(model.parnames))+len(model.parnames)]
            mod[:,k] = pskspectrum.specfit.get_model_frompars(self.scouseobject.xtrim, modparams)
        totmod = np.nansum(mod, axis=1)
        res = indivspec.spectrum[self.scouseobject.trimids]-totmod
    else:
        mod = np.zeros([len(self.scouseobject.xtrim), 1])
        totmod = np.zeros([len(self.scouseobject.xtrim), 1])
        res = indivspec.spectrum[self.scouseobject.trimids]

    return mod, res, totmod

def recreate_model_manual(self):
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

def setup_indivspec_window(self,ax,ymin=None,ymax=None):
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

    self.blank_window.text(2.75,1.20,'Select model', ha='center')
    self.blank_window.text(2.75,1.15,"Use the controls below to modify model selection", ha='center')

    return window

def plot_spectrum(self,x,y,update=False,plottoupdate=None,**kwargs):
    """
    GUI setup
    """
    if update:
        self.spectrum_window.set_xlim(np.nanmin(x),np.nanmax(x))
        plottoupdate.set_xdata(x)
        plottoupdate.set_ydata(y)
        plottoupdate.set_ls(kwargs['ls'])
        return plottoupdate
    else:
        self.spectrum_window.set_xlim(np.nanmin(x),np.nanmax(x))
        plot=self.spectrum_window.plot(x,y,**kwargs)
        return plot

def plot_snr(self,x,y,update=False,plottoupdate=None,**kwargs):
    """
    GUI setup
    """
    if update:
        plottoupdate.set_xdata(x)
        plottoupdate.set_ydata(y)
        return plottoupdate
    else:
        plot=self.spectrum_window.plot(x,y,**kwargs)
        return plot

def plot_peak_locations(self,x,y,update=False,plottoupdate=None,**kwargs):
    """
    GUI setup
    """
    if update:
        plottoupdate.set_xdata(x)
        plottoupdate.set_ydata(y)
        plottoupdate.set_markersize(kwargs['markersize'])
        return plottoupdate
    else:
        return self.spectrum_window.plot(x,y,**kwargs)

def plot_stems(self,x,y,update=False,**kwargs):
    """
    GUI setup
    """
    if update:
        for i in range(len(self.plot_peak_lines)):
            self.plot_peak_lines[i].pop(0).remove()

    self.plot_peak_lines=[]
    for i in range(np.size(x)):
        self.plot_peak_lines_indiv = self.spectrum_window.plot([x[i],x[i]],[0,y[i]],**kwargs)
        self.plot_peak_lines.append(self.plot_peak_lines_indiv)
    return self.plot_peak_lines

def plot_model(self, x, y, update=False,plottoupdate=None,**kwargs):
    """
    GUI setup
    """
    if update:
        plottoupdate.set_xdata(x)
        plottoupdate.set_ydata(y)
        return plottoupdate
    else:
        return self.spectrum_window.plot(x,y,**kwargs)

def print_information(self,xloc,yloc,str,**kwargs):
    """
    GUI setup
    """
    return self.spectrum_window.text(xloc,yloc,str,transform=self.spectrum_window.transAxes,ha='center', **kwargs)

def update_text(textobject,textstring):
    """
    GUI setup
    """
    textobject.set_text(textstring)

def make_button(ax,name,function,**kwargs):
    """
    GUI setup
    """
    from matplotlib.widgets import Button
    import matplotlib.patches as mpatches
    mybutton=Button(ax,name,**kwargs)
    mybutton.on_clicked(function)
    return mybutton

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

def update_sliders(self):
    """
    GUI setup
    """
    self.slider_vmin.valmin = self.vmin
    self.slider_vmin.valmax = self.vmax
    self.slider_vmin.valinit = self.vmin

    self.slider_vmax.valmin = self.vmin
    self.slider_vmax.valmax = self.vmax
    self.slider_vmax.valinit = self.vmax

    self.slider_vmin_ax.clear()
    self.slider_vmin=make_slider(self.slider_vmin_ax,"vmin",self.vmin,self.vmax,self.update_vmin,valinit=self.vmin,valfmt="%1.2f",color='0.75')
    self.slider_vmax_ax.clear()
    self.slider_vmax=make_slider(self.slider_vmax_ax,"vmax",self.vmin,self.vmax,self.update_vmax,valinit=self.vmax,valfmt="%1.2f",color='0.75')

def make_textbox(ax,heading,text,function,**kwargs):
    """
    GUI setup
    """
    from matplotlib.widgets import TextBox
    mytextbox=TextBox(ax,heading,initial=text, color='1')
    mytextbox.on_submit(function)
    return mytextbox
