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

class ScouseCoverage(object):
    """
    Interactive coverage selector for scouse
    """
    def __init__(self,
                 scouseobject=None):

        # For moments
        self.scouseobject=scouseobject
        self.mask_below=0.0
        self.cube=scouseobject.cube
        self.xmin = 0
        self.xmax = self.cube.shape[2]
        self.ymin = 0
        self.ymax = self.cube.shape[1]
        self.velmin = np.around(np.nanmin(self.scouseobject.x),decimals=2)
        self.velmax = np.around(np.nanmax(self.scouseobject.x),decimals=2)

        # For coverage
        self.wsaa=[3]
        self.fillfactor=[0.5]
        self.covmethod='regular'
        self.samplesize=10
        self.spacing='nyquist'
        self.speccomplexity='momdiff'

        # imports
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib import rcParams
        rcParams['font.family']= 'Arial'
        rcParams['font.size']= 9
        rcParams['lines.linewidth']= 1.     ## line width in points
        rcParams['axes.labelsize'] =18  ## fontsize of the x any y labels
        rcParams['xtick.labelsize']=16 ## fontsize of the tick labels
        rcParams['ytick.labelsize'] =16 ## fontsize of the tick labels
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

        # compute moments
        self.moments = compute_moments(self)

        plt.ioff()
        #================#
        # initiate the GUI
        #================#
        self.fig = plt.figure(figsize=(14, 8))

        #===============#
        # plot window
        #===============#
        # logo
        # self.logo_window_ax=[0.8,0.90,0.2,0.1]
        # self.logo_window=setup_plot_window(self,self.logo_window_ax)
        # self.logo_window.axis('off')
        # logo = mpimg.imread('/Users/henshaw/Dropbox/Work/Documents/Logo/SCOUSE_LOGO_FINAL-01.jpg')
        # self.logo = plt.imshow(logo,interpolation='nearest')

        # Set up the plot defaults
        self.blank_window_ax=[0.1,0.2,0.375,0.6]
        self.blank_window=setup_plot_window(self,self.blank_window_ax)
        self.map_window=setup_map_window(self)
        self.moment = 0
        self.vmin = np.nanmin(self.moments[0].value)
        self.vmax = np.nanmax(self.moments[0].value)
        # plot the moment map
        self.map = plot_map(self, self.moments[0])

        #=======#
        # sliders
        #=======#
        # create sliders for controlling the imshow limits
        self.slider_vmin_ax=self.fig.add_axes([0.1, 0.84, 0.375, 0.015])
        self.slider_vmin=make_slider(self.slider_vmin_ax,"vmin",self.vmin,self.vmax,self.update_vmin,valinit=self.vmin,valfmt="%1.2f", facecolor='0.75')
        self.slider_vmax_ax=self.fig.add_axes([0.1, 0.8125, 0.375, 0.015])
        self.slider_vmax=make_slider(self.slider_vmax_ax,"vmax",self.vmin,self.vmax,self.update_vmax,valinit=self.vmax,valfmt="%1.2f", facecolor='0.75')

        #====================#
        # compute moments menu
        #====================#
        textboxheight=0.025
        textboxwidth=0.05
        top=0.78
        mid=0.06
        space=0.025
        smallspace=space/2.

        # Create a menu bar for moment computation
        self.menu_ax=[0.025,0.2,0.07,0.6]
        self.menu=setup_plot_window(self,self.menu_ax,color='0.75')
        #elf.text_mom=self.fig.text(mid,0.78,'moments',ha='center',va='center')

        # Controls for masking
        masktop=top
        self.text_mask=self.fig.text(mid,top,'mask below',ha='center',va='center')
        self.textbox_maskbelow_ax=self.fig.add_axes([mid-textboxwidth/2., masktop-3*smallspace, textboxwidth, textboxheight])
        self.textbox_maskbelow=make_textbox(self.textbox_maskbelow_ax,'',str(self.xmin),lambda text: self.change_text(text,_type='mask'))
        maskbottom=masktop-4*smallspace

        # Controls for setting xlimits
        xlimtop=maskbottom-space
        self.text_xlim=self.fig.text(mid,xlimtop,'x limits',ha='center',va='center')
        self.textbox_xmin_ax=self.fig.add_axes([mid-textboxwidth/2., xlimtop-3*smallspace, textboxwidth, textboxheight])
        self.textbox_xmin=make_textbox(self.textbox_xmin_ax,'',str(self.xmin),lambda text: self.change_text(text,_type='xmin'))
        self.textbox_xmax_ax=self.fig.add_axes([mid-textboxwidth/2., xlimtop-6*smallspace, textboxwidth, textboxheight])
        self.textbox_xmax=make_textbox(self.textbox_xmax_ax,'',str(self.xmax),lambda text: self.change_text(text,_type='xmax'))
        xlimbottom=xlimtop-7*smallspace

        # Controls for setting ylimits
        ylimtop=xlimbottom-space
        self.text_ylim=self.fig.text(mid,ylimtop,'y limits',ha='center')
        self.textbox_ymin_ax=self.fig.add_axes([mid-textboxwidth/2., ylimtop-3*smallspace, textboxwidth, textboxheight])
        self.textbox_ymin=make_textbox(self.textbox_ymin_ax,'',str(self.ymin),lambda text: self.change_text(text,_type='ymin'))
        self.textbox_ymax_ax=self.fig.add_axes([mid-textboxwidth/2., ylimtop-6*smallspace, textboxwidth, textboxheight])
        self.textbox_ymax=make_textbox(self.textbox_ymax_ax,'',str(self.ymax),lambda text: self.change_text(text,_type='ymax'))
        ylimbottom=ylimtop-7*smallspace

        # Controls for setting vlimits
        vlimtop=ylimbottom-space
        self.text_vlim=self.fig.text(mid,vlimtop,'v limits',ha='center')
        self.textbox_vmin_ax=self.fig.add_axes([mid-textboxwidth/2., vlimtop-3*smallspace, textboxwidth, textboxheight])
        self.textbox_vmin=make_textbox(self.textbox_vmin_ax,'',str(self.velmin),lambda text: self.change_text(text,_type='velmin'))
        self.textbox_vmax_ax=self.fig.add_axes([mid-textboxwidth/2., vlimtop-6*smallspace, textboxwidth, textboxheight])
        self.textbox_vmax=make_textbox(self.textbox_vmax_ax,'',str(self.velmax),lambda text: self.change_text(text,_type='velmax'))
        vlimbottom=vlimtop-7*smallspace

        # Compute moments button
        mombuttontop=vlimbottom-space*7
        self.mom_ax=self.fig.add_axes([0.035, mombuttontop, 0.05, 2*space])
        self.mom=make_button(self,self.mom_ax,"run\nmoments",self.run_moments, color='palegreen',hovercolor='springgreen')
        mombuttonbottom=mombuttontop-3*space

        #====================#
        # plot moments menu
        #====================#
        # Controls which moment map gets plotted
        self.mom0_ax=self.fig.add_axes([0.1375, 0.14, 0.0625, 0.05])
        self.mom0=make_button(self,self.mom0_ax,"moment 0",lambda event: self.update_map(event, map=0),color='0.75',hovercolor='0.95')

        self.mom1_ax=self.fig.add_axes([0.2125, 0.14, 0.0625, 0.05])
        self.mom1=make_button(self,self.mom1_ax,"moment 1",lambda event: self.update_map(event, map=1),color='0.75',hovercolor='0.95')

        self.mom2_ax=self.fig.add_axes([0.2875, 0.14, 0.0625, 0.05])
        self.mom2=make_button(self,self.mom2_ax,"moment 2",lambda event: self.update_map(event, map=2),color='0.75',hovercolor='0.95')

        self.mom9_ax=self.fig.add_axes([0.3625, 0.14, 0.0625, 0.05])
        self.mom9=make_button(self,self.mom9_ax,"vel @ peak",lambda event: self.update_map(event, map=3),color='0.75',hovercolor='0.95')

        #==================#
        # coverage controls
        #==================#
        # Create a menu bar for moment computation
        mid = 0.515
        self.menu_ax=[0.48,0.2,0.07,0.6]
        self.menu=setup_plot_window(self,self.menu_ax,color='0.75')
        #self.text_cov=self.fig.text(mid,0.78,'coverage',ha='center',va='center')

        # Controls for saa size
        saatop=top
        self.text_saa_size=self.fig.text(mid,saatop,'SAA size',ha='center',va='center')
        self.textbox_saa_ax=self.fig.add_axes([mid-textboxwidth/2., saatop-3*smallspace, textboxwidth, textboxheight])
        self.textbox_saa=make_textbox(self.textbox_saa_ax,'',str(self.wsaa),lambda text: self.change_text(text,_type='saa'))
        saabottom=saatop-3.*smallspace

        # Controls for setting filling factor
        filltop=saabottom-space
        self.text_fill=self.fig.text(mid,filltop,'filling factor',ha='center',va='center')
        self.textbox_fill_ax=self.fig.add_axes([mid-textboxwidth/2.,filltop-3*smallspace, textboxwidth, textboxheight])
        self.textbox_fill=make_textbox(self.textbox_fill_ax,'',str(self.fillfactor),lambda text: self.change_text(text,_type='fill'))
        fillbottom=filltop-3.*smallspace

        # Controls for setting spacing
        methodtop=fillbottom-space
        self.text_method=self.fig.text(mid,methodtop,'method',ha='center',va='center')
        self.radiobutton_method_ax=self.fig.add_axes([mid-textboxwidth/2., methodtop-7*smallspace, textboxwidth, textboxheight*3.])
        self.radiobutton_method=make_radiobuttons(self.radiobutton_method_ax, ('regular','random'),self.change_covmethod,activecolor='red')
        methodbottom=methodtop-7.*smallspace

        # Controls for setting filling factor
        sampletop=methodbottom-space
        self.text_sample=self.fig.text(mid,sampletop,'sample size',ha='center',va='center')
        self.textbox_sample_ax=self.fig.add_axes([mid-textboxwidth/2.,sampletop-3*smallspace, textboxwidth, textboxheight])
        self.textbox_sample=make_textbox(self.textbox_sample_ax,'',str(self.samplesize),lambda text: self.change_text(text,_type='sample'))
        samplebottom=sampletop-3.*smallspace

        # Controls for setting spacing
        spacetop=samplebottom-space
        self.text_spacing=self.fig.text(mid,spacetop,'spacing',ha='center',va='center')
        self.radiobutton_space_ax=self.fig.add_axes([mid-textboxwidth/2., spacetop-7*smallspace, textboxwidth, textboxheight*3.])
        self.radiobutton_space=make_radiobuttons(self.radiobutton_space_ax, ('nyquist','regular'),self.change_spacing,activecolor='red')
        spacebottom=spacetop-7.*smallspace

        # Controls for spectral complexity measure
        complextop=spacebottom-space
        self.text_complex=self.fig.text(mid,complextop,'complexity',ha='center',va='center')
        self.radiobutton_complex_ax=self.fig.add_axes([mid-textboxwidth/2., complextop-7*smallspace, textboxwidth, textboxheight*3.])
        self.radiobutton_complex=make_radiobuttons(self.radiobutton_complex_ax, ('$|m_1$-$v_p|$','kurtosis'),self.change_speccomplexity,activecolor='red')
        complexbottom=complextop-7.*smallspace

        # Compute coverage button
        covbuttontop=mombuttontop
        self.cov_ax=self.fig.add_axes([0.49, covbuttontop, 0.05, 2*space])
        self.cov=make_button(self,self.cov_ax,"run\ncoverage",self.run_coverage, color='palegreen',hovercolor='springgreen')
        covbuttonbottom=covbuttontop-3*space
        #==================#
        # information window
        #==================#
        #self.information_window_ax=[0.5,0.2,0.475,0.3]
        #self.information_window=setup_plot_window(self,self.information_window_ax)

    def show(self):
        """
        Show the plot
        """
        import matplotlib.pyplot as plt
        plt.show()

    def update_map(self,event,map=None):
        """
        Controls what happens when one of the map buttons is pressed
        """
        # Get the map
        self.moment=map
        # update the limits
        self.vmin = np.nanmin(self.moments[self.moment].value)
        self.vmax = np.nanmax(self.moments[self.moment].value)
        # update the sliders
        update_sliders(self)
        # plot the map
        self.map=plot_map(self,self.moments[self.moment],update=True,plottoupdate=self.map)
        # update plot
        self.fig.canvas.draw()

    def run_moments(self,event):
        """
        Controls what happens when the compute moments button is pressed
        """
        # compute moments
        self.moments = compute_moments(self)
        # update the limits
        self.vmin = np.nanmin(self.moments[self.moment].value)
        self.vmax = np.nanmax(self.moments[self.moment].value)
        # update the sliders
        update_sliders(self)
        # plot the map
        self.map=plot_map(self,self.moments[self.moment],update=True,plottoupdate=self.map)
        # update plot
        self.fig.canvas.draw()

    def update_vmin(self,pos=None):
        """
        Controls what happens when the vmin slider is updated
        """
        # set the upper limits otherwise it'll go a bit weird
        if pos > self.vmax:
            self.vmin=self.vmax
        else:
            self.vmin = pos
        # plot the map with the new slider values
        self.map=plot_map(self,self.moments[self.moment],update=True,plottoupdate=self.map)
        # update plot
        self.fig.canvas.draw()

    def update_vmax(self,pos=None):
        """
        Controls what happens when the vmax slider is updated
        """
        # set the upper limits otherwise it'll go a bit weird
        if pos < self.vmin:
            self.vmax=self.vmin
        else:
            self.vmax=pos
        # plot the map with the new slider values
        self.map=plot_map(self,self.moments[self.moment],update=True,plottoupdate=self.map)
        # update plot
        self.fig.canvas.draw()

    def change_text(self, text, _type=None):
        """
        Controls what happens if the text boxes are updated
        """
        # extract value from text input.
        value = eval(text)

        # update value
        if _type=='xmin':
            value=int(value)
            # make sure index can't be less than 0
            if value < 0:
                value=0
            self.xmin=value
        elif _type=='xmax':
            # or greater than the length of the x axis
            if value > self.scouseobject.cube.shape[2]:
                value=self.scouseobject.cube.shape[2]
            value=int(value)
            self.xmax=value

        elif _type=='ymin':
            # make sure index can't be less than 0
            if value < 0:
                value=0
            value=int(value)
            self.ymin=value
        elif _type=='ymax':
            # or greater than the length of the y axis
            if value > self.scouseobject.cube.shape[1]:
                value=self.scouseobject.cube.shape[1]
            value=int(value)
            self.ymax=value

        elif _type=='velmin':
            # make sure index can't be less than the minimum of the vel axis
            if value < self.scouseobject.x[0]:
                value=self.scouseobject.x[0]
            else:
                self.velmin=value
        elif _type=='velmax':
            # or greater than the length of the vel axis
            if value > self.scouseobject.x[-1]:
                value = self.scouseobject.x[-1]
            else:
                self.velmax=value

        elif _type=='mask':
            self.mask_below=value

        elif _type=='saa':
            # check to see if multiple wsaa values are given
            if np.size(value)>1:
                # Create a list of wsaa values
                self.wsaa=list(value)
                # check if they are descending
                check_descending=all(earlier >= later for earlier, later in zip(self.wsaa, self.wsaa[1:]))
                # if not then sort them so that they are
                if not check_descending:
                    self.wsaa.sort(reverse=True)
            else:
                self.wsaa=value

        elif _type=='fill':
            # check to see if multiple wsaa values are given
            if np.size(self.wsaa)>1:
                # compare the size of the fillfactor list to the wsaa list
                if np.size(value)==np.size(self.wsaa):
                    # if they are the same create the fillfactor list accordingly
                    self.fillfactor=[value[i] for i in range(np.size(self.wsaa))]
                else:
                    # if not then just use the first value in the fillfactor list
                    self.fillfactor=[value[0] for i in range(np.size(self.wsaa))]
            else:
                # if only a single wsaa value is given always take the first value
                # of the fillfactor list
                self.fillfactor=[value[0] for i in range(np.size(self.wsaa))]

        elif _type=='sample':
            self.samplesize=int(value)

        else:
            pass

    def change_covmethod(self,label):
        """
        Controls what happens if coverage method radio button is clicked
        """
        self.covmethod=label

    def change_spacing(self,label):
        """
        Controls what happens if spacing radio button is clicked
        """
        self.spacing=label

    def change_speccomplexity(self,label):
        """
        Controls what happens if spectral complexity radio button is clicked
        """
        self.speccomplexity=label

    def run_coverage(self):
        pass

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
        if self.moments[0].wcs is not None:
            warnings.warn("`WCSAxes` package required for wcs coordinate display.")

    newaxis=[self.blank_window_ax[0]+0.03, self.blank_window_ax[1]+0.03, self.blank_window_ax[2]-0.06,self.blank_window_ax[3]-0.045]

    if self.cube.wcs is not None and self._wcaxes_imported:
        ax_image = WCSAxes(self.fig, newaxis, wcs=self.moments[0].wcs, slices=('x','y'))
        map_window = self.fig.add_axes(ax_image)
        x = map_window.coords[0]
        y = map_window.coords[1]
        x.set_ticklabel(exclude_overlapping=True)
        y.set_ticklabel(rotation=90,verticalalignment='bottom', horizontalalignment='left',exclude_overlapping=True)
    else:
        map_window = self.fig.add_axes(newaxis)
    return map_window

def compute_moments(self):
    """
    Create moment maps using spectral cube
    """
    from astropy import units as u
    cube=trim_cube(self)
    momzero = cube.with_mask(cube > u.Quantity(self.mask_below,cube.unit)).spectral_slab(self.velmin*u.km/u.s,self.velmax*u.km/u.s).moment0(axis=0)
    momone = cube.with_mask(cube > u.Quantity(self.mask_below,cube.unit)).spectral_slab(self.velmin*u.km/u.s,self.velmax*u.km/u.s).moment1(axis=0)
    momtwo = cube.with_mask(cube > u.Quantity(self.mask_below,cube.unit)).spectral_slab(self.velmin*u.km/u.s,self.velmax*u.km/u.s).linewidth_sigma()
    slab = cube.spectral_slab(self.velmin*u.km/u.s,self.velmax*u.km/u.s)
    maskslab = cube.with_mask(cube > u.Quantity(self.mask_below,cube.unit)).spectral_slab(self.velmin*u.km/u.s,self.velmax*u.km/u.s)

    momnine = np.empty(np.shape(momone))
    momnine.fill(np.nan)
    slabarr = np.copy(slab.unmasked_data[:].value)
    idnan = (~np.isfinite(slabarr))
    negative_inf = -1e10
    slabarr[idnan] = negative_inf
    idxmax = np.nanargmax(slabarr, axis=0)
    momnine = slab.spectral_axis[idxmax].value
    momnine[~maskslab.mask.include().any(axis=0)] = np.nan
    idnan = (np.isfinite(momtwo.value)==0)
    momnine[idnan] = np.nan
    momnine = momnine * u.km/u.s

    moments=[momzero, momone, momtwo, momnine]
    return moments

def trim_cube(self):
    """
    Trims the x,y values of the cube, returns a trimmed cube
    """
    cube = self.scouseobject.cube[:,self.ymin:self.ymax,self.xmin:self.xmax]
    return cube

def plot_map(self, map, update=False, plottoupdate=None):
    """
    map plotting
    """
    import matplotlib.pyplot as plt
    if update:
        empty=np.empty(np.shape(map.value))
        empty[:]=np.nan
        self.map.set_data(empty)

        return self.map_window.imshow(map.value, origin='lower', interpolation='nearest',cmap=plt.cm.gray, vmin=self.vmin, vmax=self.vmax)
    else:
        return self.map_window.imshow(map.value, origin='lower', interpolation='nearest',cmap=plt.cm.gray, vmin=self.vmin, vmax=self.vmax)

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

def make_radiobuttons(ax,options,function,**kwargs):
    """
    GUI setup
    """
    from matplotlib.widgets import RadioButtons
    myradiobuttons=RadioButtons(ax,options,**kwargs)
    for circle in myradiobuttons.circles: # adjust radius here. The default is 0.05
        circle.set_radius(0.05)
    myradiobuttons.on_clicked(function)
    return myradiobuttons
