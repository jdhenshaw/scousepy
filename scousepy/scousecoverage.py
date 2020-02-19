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

class ScouseCoverage(object):
    """
    Interactive coverage selector for scouse
    """
    def __init__(self,
                 scouseobject=None):

        #
        self.scouseobject=scouseobject
        self.mask_below=1.0
        self.cube=scouseobject.cube
        self.xmin = 0
        self.xmax = self.cube.shape[2]
        self.ymin = 0
        self.ymax = self.cube.shape[1]
        self.velmin = np.around(np.nanmin(self.scouseobject.x),decimals=2)
        self.velmax = np.around(np.nanmax(self.scouseobject.x),decimals=2)

        # imports
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
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
        self.blank_window_ax=[0.1,0.2,0.375,0.625]
        self.blank_window=setup_plot_window(self,self.blank_window_ax)
        self.map_window=setup_map_window(self)
        self.moment = 0
        self.vmin = np.nanmin(self.moments[0].value)
        self.vmax = np.nanmax(self.moments[0].value)
        self.map = plot_map(self, self.moments[0])

        #=======#
        # buttons
        #=======#
        self.mom_ax=self.fig.add_axes([0.2375, 0.9, 0.1, 0.05])
        self.mom=make_button(self.mom_ax,"run moments",self.run_moments)

        self.mom0_ax=self.fig.add_axes([0.1, 0.08, 0.075, 0.05])
        self.mom0=make_button(self.mom0_ax,"moment 0",lambda event: self.update_map(event, map=0))

        self.mom1_ax=self.fig.add_axes([0.2, 0.08, 0.075, 0.05])
        self.mom1=make_button(self.mom1_ax,"moment 1",lambda event: self.update_map(event, map=1))

        self.mom2_ax=self.fig.add_axes([0.3, 0.08, 0.075, 0.05])
        self.mom2=make_button(self.mom2_ax,"moment 2",lambda event: self.update_map(event, map=2))

        self.mom9_ax=self.fig.add_axes([0.4, 0.08, 0.075, 0.05])
        self.mom9=make_button(self.mom9_ax,"vel @ peak",lambda event: self.update_map(event, map=3))

        #=======#
        # sliders
        #=======#
        self.slider_vmin_ax=self.fig.add_axes([0.1, 0.17, 0.375, 0.02])
        self.slider_vmin=make_slider(self.slider_vmin_ax,"vmin",self.vmin,self.vmax,self.update_vmin,valinit=self.vmin,valfmt="%1.2f")
        self.slider_vmax_ax=self.fig.add_axes([0.1, 0.14, 0.375, 0.02])
        self.slider_vmax=make_slider(self.slider_vmax_ax,"vmax",self.vmin,self.vmax,self.update_vmax,valinit=self.vmax,valfmt="%1.2f")

        #=======#
        # ppv vol
        #=======#
        self.text_xrange=self.fig.text(0.12,0.87,'set x range')
        self.textbox_xmin_ax=self.fig.add_axes([0.1, 0.835, 0.0475, 0.025])
        self.textbox_xmin=make_textbox(self.textbox_xmin_ax,'',str(self.xmin),lambda text: self.change_text(text,_type='xmin'))

        self.textbox_xmax_ax=self.fig.add_axes([0.1525, 0.835, 0.0475, 0.025])
        self.textbox_xmax=make_textbox(self.textbox_xmax_ax,'',str(self.xmax),lambda text: self.change_text(text,_type='xmax'))

        self.text_xrange=self.fig.text(0.2575,0.87,'set y range')
        self.textbox_ymin_ax=self.fig.add_axes([0.2375, 0.835, 0.0475, 0.025])
        self.textbox_ymin=make_textbox(self.textbox_ymin_ax,'',str(self.ymin),lambda text: self.change_text(text,_type='ymin'))

        self.textbox_ymax_ax=self.fig.add_axes([0.29, 0.835, 0.0475, 0.025])
        self.textbox_ymax=make_textbox(self.textbox_ymax_ax,'',str(self.ymax),lambda text: self.change_text(text,_type='ymax'))

        self.text_xrange=self.fig.text(0.39,0.87,'set vel range')
        self.textbox_velmin_ax=self.fig.add_axes([0.375, 0.835, 0.0475, 0.025])
        self.textbox_velmin=make_textbox(self.textbox_velmin_ax,'',str(self.velmin),lambda text: self.change_text(text,_type='velmin'))

        self.textbox_velmax_ax=self.fig.add_axes([0.4275, 0.835, 0.0475, 0.025])
        self.textbox_velmax=make_textbox(self.textbox_velmax_ax,'',str(self.velmax),lambda text: self.change_text(text,_type='velmax'))

    def show(self):
        """
        Show the plot
        """
        import matplotlib.pyplot as plt
        plt.show()

    def update_map(self,event,map=None):
        """

        """
        self.moment=map
        self.vmin = np.nanmin(self.moments[self.moment].value)
        self.vmax = np.nanmax(self.moments[self.moment].value)

        update_sliders(self)

        self.map=plot_map(self,self.moments[self.moment],update=True,plottoupdate=self.map)
        # update plot
        self.fig.canvas.draw()

    def run_moments(self,event):
        """

        """
        print(self.xmin,self.xmax,self.ymin,self.ymax,self.velmin,self.velmax)
        self.moments = compute_moments(self)
        self.map=plot_map(self,self.moments[self.moment],update=True,plottoupdate=self.map)
        # update plot
        self.fig.canvas.draw()

    def update_vmin(self,pos=None):
        """

        """
        if pos > self.vmax:
            self.vmin=self.vmax
        else:
            self.vmin = pos
        self.map=plot_map(self,self.moments[self.moment])
        # update plot
        self.fig.canvas.draw()

    def update_vmax(self,pos=None):
        """

        """
        if pos < self.vmin:
            self.vmax=self.vmin
        else:
            self.vmax=pos

        self.map=plot_map(self,self.moments[self.moment])
        # update plot
        self.fig.canvas.draw()

    def change_text(self, text, _type=None):
        """

        """
        # extract value from text input.
        value = eval(text)
        # update value
        if _type=='xmin':
            value=int(value)
            if value < 0:
                value=0
            self.xmin=value
        elif _type=='xmax':
            if value > self.scouseobject.cube.shape[2]:
                value=self.scouseobject.cube.shape[2]
            value=int(value)
            self.xmax=value
        elif _type=='ymin':
            if value < 0:
                value=0
            value=int(value)
            self.ymin=value
        elif _type=='ymax':
            if value > self.scouseobject.cube.shape[1]:
                value=self.scouseobject.cube.shape[1]
            value=int(value)
            self.ymax=value
        elif _type=='velmin':
            self.velmin=value
        elif _type=='velmax':
            self.velmax=value
        else:
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
    Create moment maps

    Parameters
    ----------
    scouseobject : Instance of the scousepy class
    write_moments : bool
        if True scouse will output the moments as fits files
    dir : string
        Output directory for moment files
    filename : string
        output file name for moment files (scousepy will add suffix relating to
        the different moments)
    verbose : bool
        verbose output

    """
    from astropy import units as u
    cube=trim_cube(self)
    momzero = cube.with_mask(cube > u.Quantity(self.scouseobject.mask_below,cube.unit)).spectral_slab(self.velmin*u.km/u.s,self.velmax*u.km/u.s).moment0(axis=0)
    momone = cube.with_mask(cube > u.Quantity(self.scouseobject.mask_below,cube.unit)).spectral_slab(self.velmin*u.km/u.s,self.velmax*u.km/u.s).moment1(axis=0)
    momtwo = cube.with_mask(cube > u.Quantity(self.scouseobject.mask_below,cube.unit)).spectral_slab(self.velmin*u.km/u.s,self.velmax*u.km/u.s).linewidth_sigma()
    slab = cube.spectral_slab(self.velmin*u.km/u.s,self.velmax*u.km/u.s)
    maskslab = cube.with_mask(cube > u.Quantity(self.scouseobject.mask_below,cube.unit)).spectral_slab(self.velmin*u.km/u.s,self.velmax*u.km/u.s)

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

    """
    cube = self.scouseobject.cube[:,self.ymin:self.ymax,self.xmin:self.xmax]
    return cube

def plot_map(self, map, update=False, plottoupdate=None):
    """

    """
    import matplotlib.pyplot as plt
    if update:
        #self.map_window.clear()
        #self.map_window=setup_map_window(self)
        return self.map_window.imshow(map.value, origin='lower', interpolation='nearest',cmap=plt.cm.gray, vmin=self.vmin, vmax=self.vmax)
    else:
        return self.map_window.imshow(map.value, origin='lower', interpolation='nearest',cmap=plt.cm.gray, vmin=self.vmin, vmax=self.vmax)

def make_button(ax,name,function,**kwargs):
    """
    GUI setup
    """
    from matplotlib.widgets import Button
    mybutton=Button(ax,name)
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
    self.slider_vmin=make_slider(self.slider_vmin_ax,"vmin",self.vmin,self.vmax,self.update_vmin,valinit=self.vmin,valfmt="%1.2f")
    self.slider_vmax_ax.clear()
    self.slider_vmax=make_slider(self.slider_vmax_ax,"vmax",self.vmin,self.vmax,self.update_vmax,valinit=self.vmax,valfmt="%1.2f")

def make_textbox(ax,heading,text,function,**kwargs):
    """
    GUI setup
    """
    from matplotlib.widgets import TextBox
    mytextbox=TextBox(ax,heading,initial=text, color='1')
    mytextbox.on_submit(function)
    return mytextbox
