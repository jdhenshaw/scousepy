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

class ScouseFitChecker(object):
    """
    Interactive model inspector for scouse

    Parameters
    ----------
    scouseobject : scouse class object
        Instance of the scouse object.

    """
    def __init__(self, scouseobject=None, verbose=True, blocksize=5., selected_spectra=None):

        self.scouseobject=scouseobject
        self.cube=scouseobject.cube
        self.verbose=verbose
        self.blocksize=int(blocksize)
        self.xpos=None
        self.ypos=None
        self.keys=None
        self.check_spec_indices=[]
        self.selected_spectra=selected_spectra

        # imports
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib import rcParams

        rcParams['font.family']= 'Arial'
        rcParams['font.size']= 9
        rcParams['lines.linewidth']= 1.     ## line width in points
        rcParams['axes.labelsize'] =18  ## fontsize of the x any y labels
        rcParams['xtick.labelsize']=5 ## fontsize of the tick labels
        rcParams['ytick.labelsize'] =5 ## fontsize of the tick labels
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

        # compute diagnostics
        self.diagnostics = compute_diagnostic_plots(self)

        #================#
        # initiate the GUI
        #================#
        self.fig = plt.figure(figsize=(14, 8))

        #===============#
        # plot window
        #===============#

        # Set up the plot defaults
        self.blank_window_ax=[0.1,0.2,0.375,0.6]
        self.blank_window=setup_plot_window(self,self.blank_window_ax)
        self.map_window=setup_map_window(self)
        self.spec_window_ax=[0.5,0.2,0.375,0.6]
        self.spec_window_axes=setup_spec_window(self)

        self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.fig.canvas.mpl_connect('key_press_event', self.keyentry)

        self.diagnostic=0
        self.vmin=np.nanmin(self.diagnostics[0])-0.05*np.nanmin(self.diagnostics[0])
        self.vmax=np.nanmax(self.diagnostics[0])+0.05*np.nanmax(self.diagnostics[0])
        # plot the diagnostics
        self.cmap=get_cmap(self)
        self.map=plot_map(self, self.diagnostics[0])

        #=======#
        # sliders
        #=======#
        # create sliders for controlling the imshow limits
        self.slider_vmin_ax=self.fig.add_axes([0.1, 0.84, 0.375, 0.015])
        self.slider_vmin=make_slider(self.slider_vmin_ax,"vmin",self.vmin,self.vmax,self.update_vmin,valinit=self.vmin,valfmt="%1.2f", facecolor='0.75')
        self.slider_vmax_ax=self.fig.add_axes([0.1, 0.8125, 0.375, 0.015])
        self.slider_vmax=make_slider(self.slider_vmax_ax,"vmax",self.vmin,self.vmax,self.update_vmax,valinit=self.vmax,valfmt="%1.2f", facecolor='0.75')

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
        space=0.015
        size=0.05

        # Controls which moment map gets plotted
        start=0.1
        self.diag0_ax=self.fig.add_axes([start, 0.14, size, 0.05])
        self.diag0=make_button(self,self.diag0_ax,"rms",lambda event: self.update_map(event, map=0),color='0.75',hovercolor='0.95')
        end=start+size+space

        start=end
        self.diag1_ax=self.fig.add_axes([start, 0.14, size, 0.05])
        self.diag1=make_button(self,self.diag1_ax,"residstd",lambda event: self.update_map(event, map=1),color='0.75',hovercolor='0.95')
        end=start+size+space

        start=end
        self.diag2_ax=self.fig.add_axes([start, 0.14, size, 0.05])
        self.diag2=make_button(self,self.diag2_ax,"redchisq",lambda event: self.update_map(event, map=2),color='0.75',hovercolor='0.95')
        end=start+size+space

        start=end
        self.diag3_ax=self.fig.add_axes([start, 0.14, size, 0.05])
        self.diag3=make_button(self,self.diag3_ax,"ncomps",lambda event: self.update_map(event, map=3),color='0.75',hovercolor='0.95')
        end=start+size+space

        start=end
        self.diag4_ax=self.fig.add_axes([start, 0.14, size, 0.05])
        self.diag4=make_button(self,self.diag4_ax,"aic",lambda event: self.update_map(event, map=4),color='0.75',hovercolor='0.95')
        end=start+size+space

        start=end
        self.diag5_ax=self.fig.add_axes([start, 0.14, size, 0.05])
        self.diag5=make_button(self,self.diag5_ax,"chisq",lambda event: self.update_map(event, map=5),color='0.75',hovercolor='0.95')
        end=start+size+space

        #================
        # Continue button
        #================
        self.continue_ax=self.fig.add_axes([end+0.36, 0.14, 0.05, 0.05])
        self.continue_button=make_button(self,self.continue_ax,"continue",self.check_complete, color='lightblue',hovercolor='aliceblue')

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
                print(colors.fg._yellow_+"Warning: No spectra selected.  "+colors._endc_)
                print('')
        else:
            if self.verbose:
                print(colors.fg._lightgreen_+"Checking complete. {} spectra selected for further inspection".format(np.size(self.check_spec_indices))+colors._endc_)
                print('')
        # close the window
        self.close_window()

    def click(self, event):
        """
        Controls what happens when the interactive plot window is clicked
        """
        # create a list containing all axes
        axislist=[self.map_window]+self.spec_window_axes
        if self.fig.canvas.manager.toolbar._active is None:

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
                        self.plot_spectra()

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
        axislist=[self.map_window]+self.spec_window_axes
        if self.fig.canvas.manager.toolbar._active is None:

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
                        self.plot_spectra()

                # else if one of the spectra are selected then store the
                # information and identify the spectrum as one to be
                # checked during stage 6
                elif axisNr in np.arange(1,np.size(axislist)):
                    self.select_spectra(event, axisNr)

                else:
                    pass
                self.fig.canvas.draw()

    def plot_spectra(self):
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
                ax=self.spec_window_axes[j+i*self.blocksize]

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
                    spectrum=self.scouseobject.cube.filled_data[self.scouseobject.trimids,index[0],index[1]].value
                    # redefine the axis limits and plot the spectrum
                    ax.set_xlim(np.min(self.scouseobject.xtrim), np.max(self.scouseobject.xtrim))
                    ax.set_ylim(np.nanmin(spectrum), 1.05*np.nanmax(spectrum))
                    ax.plot(self.scouseobject.xtrim,spectrum, drawstyle='steps', color='k', lw=0.85)

                    # now check to see if a model is available
                    if key in self.scouseobject.indiv_dict.keys():
                        indivspec=self.scouseobject.indiv_dict[key]
                        # recreate the model
                        mod, res, totmod=recreate_model(self,indivspec)
                        for k in range(np.shape(mod)[1]):
                            # plot individual components
                            ax.plot(self.scouseobject.xtrim,mod[:,k], color='limegreen', lw=1)

    def select_spectra(self, event, axisNr):
        """
        Controls what happens when a spectrum is selected
        """

        if self.keys is not None:
            key=self.keys[axisNr-1]
            ax=self.spec_window_axes[axisNr-1]

            if event.name=='button_press_event':

                # select
                if event.button==1:
                    if ~np.isnan(key):
                        ax.patch.set_facecolor('red')
                        ax.patch.set_alpha(0.1)
                        if key not in self.check_spec_indices:
                            self.check_spec_indices.append(key)

                # deselect
                elif event.button==3:
                    if ~np.isnan(key):
                         ax.patch.set_facecolor('white')
                         ax.patch.set_alpha(0.0)
                         if key in self.check_spec_indices:
                             self.check_spec_indices.remove(key)

                else:
                    return

            elif event.name=='key_press_event':

                # select
                if (event.key=='enter'):
                    if ~np.isnan(key):
                        ax.patch.set_facecolor('red')
                        ax.patch.set_alpha(0.1)
                        if key not in self.check_spec_indices:
                            self.check_spec_indices.append(key)

                # deselect
                elif (event.key=='d') or (event.key=='r'):
                    if ~np.isnan(key):
                         ax.patch.set_facecolor('white')
                         ax.patch.set_alpha(0.0)
                         if key in self.check_spec_indices:
                             self.check_spec_indices.remove(key)

                # select all
                elif (event.key=='a'):
                    for axid, _key in enumerate(self.keys):
                        if ~np.isnan(_key):
                            self.spec_window_axes[axid].patch.set_facecolor('red')
                            self.spec_window_axes[axid].patch.set_alpha(0.1)
                            if _key not in self.check_spec_indices:
                                self.check_spec_indices.append(_key)

                # remove all
                elif (event.key=='backspace') or (event.key=='escape'):
                    for axid, _key in enumerate(self.keys):
                        if ~np.isnan(_key):
                            self.spec_window_axes[axid].patch.set_facecolor('white')
                            self.spec_window_axes[axid].patch.set_alpha(0.0)
                            if _key in self.check_spec_indices:
                                self.check_spec_indices.remove(_key)

                else:
                    return

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

def get_cmap(self):
    import matplotlib.pyplot as plt
    if self.diagnostic==3:
        self.cmap=plt.cm.viridis
        self.cmap.set_bad(color='lightgrey')
        self.cmap.set_under('w')
    else:
        self.cmap=plt.cm.viridis
        self.cmap.set_bad(color='lightgrey')


def compute_diagnostic_plots(self,
                             maps=['rms','residstd','redchisq','ncomps','AIC','chisq']):
    """
    Create diagnostic maps

    Parameters
    ----------
    maps : list
        a list of map names
    """
    from astropy import units as u
    from .verbose_output import print_to_terminal

    # we output the maps as fits files so first check if they exist. If they
    # do then load them rather than create them twice
    if os.path.exists(self.scouseobject.outputdirectory+self.scouseobject.filename+'/stage_5/stage_5_'+maps[0]+'.fits'):
        if self.verbose:
            progress_bar = print_to_terminal(stage='s5', step='diagnosticsload')
        diagnostics=load_maps(self,maps)
    else:
        if self.verbose:
            progress_bar = print_to_terminal(stage='s5', step='diagnosticsinit')
        diagnostics=[generate_2d_parametermap(self, mapname) for mapname in maps]
        save_maps(self,maps, diagnostics)
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
        progress_bar = print_to_terminal(stage='s5', step='diagnostics',length=len(self.scouseobject.indiv_dict.keys()))

    for key,spectrum in self.scouseobject.indiv_dict.items():
        cx,cy = spectrum.coordinates
        #if getattr(spectrum.model, 'ncomps') != 0:
        map[cy, cx] = getattr(spectrum.model, spectrum_parameter)
        #else:
        #    map[cy, cx] = np.nan
        if self.verbose:
            progress_bar.update()

    return map

def save_maps(self, maps, diagnostics, overwrite=True):
    """
    Procedure to save the output maps as fits files

    Parameters
    ----------
    maps : list
        list of the map names
    diagnostics : list
        a list containing the diagnostic plots
    overwrite : bool
        optional overwrite

    """
    from astropy.io import fits
    savedir=self.scouseobject.outputdirectory+self.scouseobject.filename+'/stage_5/'
    for index,mapname in enumerate(maps):
        fh = fits.PrimaryHDU(data=diagnostics[index], header=self.scouseobject.cube[0,:,:].header)
        fh.writeto(os.path.join(savedir, "stage_5_"+mapname+".fits"), overwrite=overwrite)

def load_maps(self, maps):
    """
    Procedure to load the maps

    Parameters
    ----------
    maps : list
        list of map names

    """
    from astropy.io import fits
    savedir=self.scouseobject.outputdirectory+self.scouseobject.filename+'/stage_5/'

    return [fits.getdata(savedir+'stage_5_'+mapname+'.fits') for mapname in maps]

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
    return window

def setup_map_window(self):
    """
    Setup the map window for plotting
    """
    try:
        from wcsaxes import WCSAxes
        self._wcaxes_imported = True
    except ImportError:
        self._wcaxes_imported = False
        if self.cube.wcs is not None:
            warnings.warn("`WCSAxes` package required for wcs coordinate display.")

    newaxis=[self.blank_window_ax[0]+0.03, self.blank_window_ax[1]+0.03, self.blank_window_ax[2]-0.06,self.blank_window_ax[3]-0.045]

    if self.cube.wcs is not None and self._wcaxes_imported:
        wcs=self.cube.wcs
        wcs=wcs.dropaxis(2)

        ax_image = WCSAxes(self.fig, newaxis, wcs=wcs, slices=('x','y'))
        map_window = self.fig.add_axes(ax_image)
        x = map_window.coords[0]
        y = map_window.coords[1]
        x.set_ticklabel(exclude_overlapping=True)
        y.set_ticklabel(rotation=90,verticalalignment='bottom', horizontalalignment='left',exclude_overlapping=True)
    else:
        map_window = self.fig.add_axes(newaxis)
    return map_window

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

    return axlist

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

def recreate_model(self, indivspec):
    """
    Recreates model from parameters in modeldict

    """
    import pyspeckit
    from scousepy.SpectralDecomposer import Decomposer

    model=indivspec.model

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

def update_text(textobject,textstring):
    """
    GUI setup
    """
    textobject.set_text(textstring)

def make_button(self,ax,name,function,**kwargs):
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
