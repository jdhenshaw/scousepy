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

    Parameters
    ----------
    scouseobject : scouse class object
        Instance of the scouse object.
    create_config_file : Bool
        Creates an astropy table containing the coverage information

    """
    def __init__(self, scouseobject=None, create_config_file=True, verbose=True, interactive=True):

        # For moments
        self.scouseobject=scouseobject
        self.verbose=verbose
        self.interactive=interactive

        # config file location
        from .io import import_from_config
        config_filename_coverage='coverage.config'
        scousedir=os.path.join(self.scouseobject.outputdirectory, self.scouseobject.filename)
        configdir=os.path.join(scousedir+'/config_files')
        configpath_coverage=os.path.join(scousedir+'/config_files', config_filename_coverage)

        # check to see if the config file exists. If it does load in the params
        # if not then set some defaults
        if os.path.exists(configpath_coverage):
            import_from_config(self, configpath_coverage)
            # Set these manually
            if (self.x_range[0] is None) or (self.x_range[0] < 0):
                self.xmin = 0
            else:
                self.xmin = self.x_range[0]
            if (self.y_range[0] is None) or (self.y_range[0] < 0):
                self.ymin = 0
            else:
                self.ymin = self.y_range[0]
            if (self.vel_range[0] is None) or (self.vel_range[0] < self.scouseobject.cube.spectral_axis[0].value):
                self.velmin = np.around(np.nanmin(self.scouseobject.cube.spectral_axis.value),decimals=2)
            else:
                self.velmin = self.vel_range[0]

            if (self.x_range[1] is None) or (self.x_range[1] > self.scouseobject.cube.shape[2]):
                self.xmax = self.scouseobject.cube.shape[2]
            else:
                self.xmax = self.x_range[1]
            if (self.y_range[1] is None) or (self.y_range[1] > self.scouseobject.cube.shape[1]):
                self.ymax = self.scouseobject.cube.shape[1]
            else:
                self.ymax = self.y_range[1]
            if (self.vel_range[1] is None) or (self.vel_range[1] > self.scouseobject.cube.spectral_axis[-1].value):
                self.velmax = np.around(np.nanmax(self.scouseobject.cube.spectral_axis.value),decimals=2)
            else:
                self.velmax = self.vel_range[1]

        else:
            self.nrefine=1
            self.mask_below=0.0
            self.mask_coverage=None
            self.xmin = 0
            self.xmax = self.scouseobject.cube.shape[2]
            self.ymin = 0
            self.ymax = self.scouseobject.cube.shape[1]
            self.velmin = np.around(np.nanmin(self.scouseobject.cube.spectral_axis.value),decimals=2)
            self.velmax = np.around(np.nanmax(self.scouseobject.cube.spectral_axis.value),decimals=2)
            self.wsaa=[3]
            self.fillfactor=[0.5]
            self.samplesize=10
            self.covmethod='regular'
            self.spacing='nyquist'
            self.speccomplexity='momdiff'
            self.totalsaas=None
            self.totalspec=None

        if np.size(self.wsaa)>1:
            self.refine_grid=True
        else:
            self.refine_grid=False

        if self.mask_coverage==None:
            self.mask_provided=False
            self._mask_found=False
        else:
            self.mask_provided=True
            self.user_mask=get_mask(self)

        # For coverage
        self.spacingvalue=None
        self.coverage=[]
        self.coverage_path=[]
        self.coverage_map=None
        self.create_config_file=create_config_file
        self.config_file=None
        self.sortedids=[0]

        # compute moments
        self.moments = compute_moments(self)
        # compute measures of spectral complexity
        self.complexity_maps = compute_spectral_complexity(self)

        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib import rcParams
        self.cmap=plt.cm.binary_r
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
        rcParams['xtick.direction']='in'
        rcParams['ytick.direction']='in'

        # remove some matplotlib keyboard shortcuts to prevent meltdown
        if 'q' in plt.rcParams['keymap.quit']:
            plt.rcParams['keymap.quit'].remove('q')
        if 'Q' in plt.rcParams['keymap.quit_all']:
            plt.rcParams['keymap.quit_all'].remove('Q')

        plt.ioff()
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
        self.moment=0
        self.vmin=np.nanmin(self.moments[0].value)-0.05*np.nanmin(self.moments[0].value)
        self.vmax=np.nanmax(self.moments[0].value)+0.05*np.nanmax(self.moments[0].value)
        # plot the moment map
        self.map=plot_map(self, self.moments[0].value)

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
        self.textbox_maskbelow=make_textbox(self.textbox_maskbelow_ax,'',str(self.mask_below),lambda text: self.change_text(text,_type='mask'))
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

        # Controls for spectral complexity measure
        cmaptop=vlimbottom-space
        self.text_cmap=self.fig.text(mid,cmaptop,'colour map',ha='center',va='center')
        self.radiobutton_cmap_ax=self.fig.add_axes([mid-textboxwidth/2., cmaptop-7*smallspace, textboxwidth, textboxheight*3.])
        self.radiobutton_cmap=make_radiobuttons(self.radiobutton_cmap_ax, ('binary','viridis','bwr'),self.change_cmap,activecolor='black')
        cmapbottom=cmaptop-7.*smallspace

        # Compute moments button
        mombuttontop=cmapbottom-space*3
        self.mom_ax=self.fig.add_axes([0.035, mombuttontop, 0.05, 2*space])
        self.mom=make_button(self,self.mom_ax,"run\nmoments",self.run_moments, color='palegreen',hovercolor='springgreen')
        mombuttonbottom=mombuttontop-3*space

        #====================#
        # plot moments menu
        #====================#
        space=0.015
        size=0.05

        # Controls which moment map gets plotted
        start=0.1
        self.mom0_ax=self.fig.add_axes([start, 0.14, size, 0.05])
        self.mom0=make_button(self,self.mom0_ax,"moment 0",lambda event: self.update_map(event, map=0),color='0.75',hovercolor='0.95')
        end=start+size+space

        start=end
        self.mom1_ax=self.fig.add_axes([start, 0.14, size, 0.05])
        self.mom1=make_button(self,self.mom1_ax,"moment 1",lambda event: self.update_map(event, map=1),color='0.75',hovercolor='0.95')
        end=start+size+space

        start=end
        self.mom2_ax=self.fig.add_axes([start, 0.14, size, 0.05])
        self.mom2=make_button(self,self.mom2_ax,"moment 2",lambda event: self.update_map(event, map=2),color='0.75',hovercolor='0.95')
        end=start+size+space

        # start=end
        # self.mom3_ax=self.fig.add_axes([start, 0.14, size, 0.05])
        # self.mom3=make_button(self,self.mom3_ax,"moment 3",lambda event: self.update_map(event, map=3),color='0.75',hovercolor='0.95')
        # end=start+size+space
        #
        # start=end
        # self.mom4_ax=self.fig.add_axes([start, 0.14, size, 0.05])
        # self.mom4=make_button(self,self.mom4_ax,"moment 4",lambda event: self.update_map(event, map=4),color='0.75',hovercolor='0.95')
        # end=start+size+space

        start=end
        self.mom9_ax=self.fig.add_axes([start, 0.14, size, 0.05])
        self.mom9=make_button(self,self.mom9_ax,"vel @ peak",lambda event: self.update_map(event, map=5),color='0.75',hovercolor='0.95')
        end=start+size+space

        start=end
        end=start+size+space

        start=end
        self.mommask_ax=self.fig.add_axes([start, 0.14, size, 0.05])
        self.mommask=make_button(self,self.mommask_ax,"mask",lambda event: self.update_map(event, map=6), color='lightblue',hovercolor='aliceblue')
        end=start+size+space

        #==================#
        # coverage controls
        #==================#
        space=0.025
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
        self.radiobutton_method=make_radiobuttons(self.radiobutton_method_ax, ('regular','random'),self.change_covmethod,activecolor='black')
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
        self.radiobutton_space=make_radiobuttons(self.radiobutton_space_ax, ('nyquist','regular'),self.change_spacing,activecolor='black')
        spacebottom=spacetop-7.*smallspace

        # Controls for spectral complexity measure
        complextop=spacebottom-space
        self.text_complex=self.fig.text(mid,complextop,'complexity',ha='center',va='center')
        self.radiobutton_complex_ax=self.fig.add_axes([mid-textboxwidth/2., complextop-7*smallspace, textboxwidth, textboxheight*3.])
        self.radiobutton_complex=make_radiobuttons(self.radiobutton_complex_ax, ('$|m_1$-$v_p|$','kurtosis'),self.change_speccomplexity,activecolor='black')
        complexbottom=complextop-7.*smallspace

        # Compute coverage button
        covbuttontop=mombuttontop
        self.cov_ax=self.fig.add_axes([0.49, covbuttontop, 0.05, 2*space])
        self.cov=make_button(self,self.cov_ax,"run\ncoverage",self.run_coverage, color='palegreen',hovercolor='springgreen')
        covbuttonbottom=covbuttontop-3*space

        #==================#
        # information window
        #==================#
        self.information_window_ax=[0.575,0.3,0.35,0.4]
        self.information_window=setup_plot_window(self,self.information_window_ax)
        # print information relating to moment analysis
        self.information_window.text(0.02, 0.94, 'moment information:',transform=self.information_window.transAxes,fontsize=10, fontweight='bold')
        self.text_mask = print_information(self,0.02,0.88,'data masked below: '+str(self.mask_below), fontsize=10)
        self.text_ppv = print_information(self,0.02,0.82,'PPV volume ([x,y,v]): [['+str(self.xmin)+', '+str(self.xmax)+'], ['+str(self.ymin)+', '+str(self.ymax)+'], ['+str(self.velmin)+', '+str(self.velmax)+']]', fontsize=10)
        # print information relating to coverage analysis
        self.information_window.text(0.02, 0.7, 'coverage information:',transform=self.information_window.transAxes,fontsize=10, fontweight='bold')
        self.text_saa = print_information(self,0.02,0.64,'SAA size(s): '+str(self.wsaa), fontsize=10)
        self.text_fillingfactor = print_information(self,0.02,0.58,'filling factor(s): '+str(self.fillfactor), fontsize=10)
        self.text_method = print_information(self,0.02,0.52,'method: '+str(self.covmethod), fontsize=10)
        self.text_spacing = print_information(self,0.02,0.46,'spacing: '+str(self.spacing), fontsize=10)
        self.text_complexity = print_information(self,0.02,0.4,'', fontsize=10)
        if self.refine_grid:
            if self.speccomplexity=='momdiff':
                update_text(self.text_complexity, 'complexity measure: $|m_1$-$v_p|$')
            elif self.speccomplexity=='kurtosis':
                print('')
                print(colors.fg._yellow_+"Warning: Kurtosis option is not available yet. Setting to default.  "+colors._endc_)
                print('')
                #update_text(self.text_complexity, 'complexity measure: kurtosis')
                update_text(self.text_complexity, 'complexity measure: $|m_1$-$v_p|$')
            else:
                update_text(self.text_complexity, 'complexity measure: '+str(self.speccomplexity))
        else:
            update_text(self.text_complexity,'')
        # prepare summary information
        self.information_window.text(0.02, 0.28, 'summary:',transform=self.information_window.transAxes,fontsize=10, fontweight='bold')
        if np.size(self.coverage)==0:
            self.text_runcoverage = print_information(self,0.17,0.28,'select run coverage', fontsize=10, color='green',ha='left')
        self.text_totalindivsaas = print_information(self,0.02,0.22,'', fontsize=10)
        self.text_totalindivspec = print_information(self,0.02,0.16,'', fontsize=10)
        self.text_summary = print_information(self,0.02,0.04,'', fontsize=10,fontweight='bold')

        #================
        # Continue button
        #================
        self.continue_ax=self.fig.add_axes([0.875, 0.24, 0.05, 0.05])
        self.continue_button=make_button(self,self.continue_ax,"continue",self.coverage_complete, color='lightblue',hovercolor='aliceblue')

        if not self.interactive:
            # run the coverage but do not display the plot
            self.run_coverage(None)
            # complete the coverage task
            self.coverage_complete(None)


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
        plt.close(self.fig)

    def coverage_complete(self, event):
        """
        Controls what happens if continue button is pressed.

        Parameters
        ----------
        event : button press event

        """
        if self.coverage_map is None:
            self.run_coverage(event)
            if self.verbose:
                print(colors.fg._yellow_+"Warning: Running coverage with default settings.  "+colors._endc_)
                print('')
        else:
            if self.verbose:
                print(colors.fg._lightgreen_+"Coverage complete. "+colors._endc_)
                print('')

        if self.create_config_file:
            if self.covmethod=='regular':
                self.samplesize=0.0
            self.config_file=make_config_file(self)

        # close the window
        self.close_window()

    def update_map(self,event,map=None):
        """
        Controls what happens when one of the map buttons is pressed

        Parameters
        ----------
        event : button press event
        map : number
            Index for the self.moments list - indicates which map to plot
        """
        # Get the map
        self.moment=map
        if (map==3) or (map==4) or (map==6):
            maptoplot=self.moments[self.moment]
        else:
            maptoplot=self.moments[self.moment].value
        # update the limits
        self.vmin = np.nanmin(maptoplot)-0.05*np.nanmin(maptoplot)
        self.vmax = np.nanmax(maptoplot)+0.05*np.nanmax(maptoplot)
        # update the sliders
        update_sliders(self)
        # plot the map
        self.map=plot_map(self,maptoplot,update=True)
        # update plot
        self.fig.canvas.draw()

    def run_moments(self,event):
        """
        Controls what happens when the compute moments button is pressed

        Parameters
        ----------
        event : button press event
        """
        # compute moments
        self.moments = compute_moments(self)
        if (self.moment==3) or (self.moment==4) or (self.moment==6):
            maptoplot=self.moments[self.moment]
        else:
            maptoplot=self.moments[self.moment].value
        # compute measures of spectral complexity
        self.complexity_maps = compute_spectral_complexity(self)
        # update the limits
        self.vmin = np.nanmin(maptoplot)-0.05*np.nanmin(maptoplot)
        self.vmax = np.nanmax(maptoplot)+0.05*np.nanmax(maptoplot)
        # update the sliders
        update_sliders(self)
        # plot the map
        self.map=plot_map(self,maptoplot,update=True)
        # update the information window
        update_text(self.text_mask, 'data masked below: '+str(self.mask_below))
        update_text(self.text_ppv, 'PPV volume ([x,y,v]): [['+str(self.xmin)+', '+str(self.xmax)+'], ['+ str(self.ymin)+', '+str(self.ymax)+'], ['+str(self.velmin)+', '+str(self.velmax)+']]')
        update_text(self.text_runcoverage, 'run coverage')
        update_text(self.text_totalindivsaas, '')
        update_text(self.text_totalindivspec, '')
        update_text(self.text_summary, '')

        # update plot
        self.fig.canvas.draw()

    def update_vmin(self,pos=None):
        """
        Controls what happens when the vmin slider is updated

        Parameters
        ----------
        pos : slider position
        """
        # set the upper limits otherwise it'll go a bit weird
        if pos > self.vmax:
            self.vmin=self.vmax
        else:
            self.vmin = pos

        if (self.moment==3) or (self.moment==4):
            maptoplot=self.moments[self.moment]
        else:
            maptoplot=self.moments[self.moment].value

        # plot the map with the new slider values
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
        # set the upper limits otherwise it'll go a bit weird
        if pos < self.vmin:
            self.vmax=self.vmin
        else:
            self.vmax=pos

        if (self.moment==3) or (self.moment==4):
            maptoplot=self.moments[self.moment]
        else:
            maptoplot=self.moments[self.moment].value

        # plot the map with the new slider values
        self.map=plot_map(self,maptoplot,update=True)
        # update plot
        self.fig.canvas.draw()

    def change_text(self, text, _type=None):
        """
        Controls what happens if the text boxes are updated

        Parameters
        ----------
        text : string
            the text within the text box
        _type : string
            indicates which textbox has been updated

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
            if value < self.scouseobject.cube.spectral_axis[0].value:
                value=self.scouseobject.cube.spectral_axis[0].value
            else:
                self.velmin=value
        elif _type=='velmax':
            # or greater than the length of the vel axis
            if value > self.scouseobject.cube.spectral_axis[-1].value:
                value = self.scouseobject.cube.spectral_axis[-1].value
            else:
                self.velmax=value

        elif _type=='mask':
            self.mask_below=value

        elif _type=='saa':
            # Always make sure it is a list
            if isinstance(value, int):
                value=[value]

            # check to see if multiple wsaa values are given
            if np.size(value)>1:
                # Create a list of wsaa values
                self.wsaa=list(value)

                # check if they are descending
                check_descending=all(earlier >= later for earlier, later in zip(self.wsaa, self.wsaa[1:]))

                self.fillfactor=[self.fillfactor[i] for i in self.sortedids]
                # if not then sort them so that they are
                if not check_descending:
                    self.sortedids=sorted(range(np.size(self.wsaa)), key=lambda k: self.wsaa[k], reverse=True)
                    self.wsaa.sort(reverse=True)
                else:
                    self.sortedids=range(np.size(self.wsaa))

                # compare the size of the fillfactor list to the wsaa list
                if np.size(self.wsaa)==np.size(self.fillfactor):
                    # if they are the same create the fillfactor list accordingly
                    self.fillfactor=[self.fillfactor[i] for i in self.sortedids]
                else:
                    # if not then just use the first value in the fillfactor list
                    self.fillfactor=[self.fillfactor[0] for i in range(np.size(self.wsaa))]

            else:
                self.wsaa=value

            if np.size(self.wsaa)>1:
                self.refine_grid=True
            else:
                self.refine_grid=False

        elif _type=='fill':
            if isinstance(value, int) or isinstance(value, float):
                value=[value]
            # check to see if multiple wsaa values are given
            if np.size(self.wsaa)>1:
                # compare the size of the fillfactor list to the wsaa list
                if np.size(value)==np.size(self.wsaa):
                    self.fillfactor=[value[i] for i in self.sortedids]
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

    def change_cmap(self,label):
        """
        Controls what happens if coverage method radio button is clicked

        Parameters
        ----------
        label : radio button label
        """
        import matplotlib.pyplot as plt
        if label=='binary':
            self.cmap=plt.cm.binary_r
        elif label=='viridis':
            self.cmap=plt.cm.viridis
        elif label=='bwr':
            self.cmap=plt.cm.bwr
        else:
            self.cmap=plt.cm.binary_r

        self.map.set_cmap(self.cmap)
        self.fig.canvas.draw()

    def change_covmethod(self,label):
        """
        Controls what happens if coverage method radio button is clicked

        Parameters
        ----------
        label : radio button label
        """
        self.covmethod=label

    def change_spacing(self,label):
        """
        Controls what happens if spacing radio button is clicked

        Parameters
        ----------
        label : radio button label
        """
        self.spacing=label

    def change_speccomplexity(self,label):
        """
        Controls what happens if spectral complexity radio button is clicked

        Parameters
        ----------
        label : radio button label
        """
        if label=='$|m_1$-$v_p|$':
            self.speccomplexity='momdiff'
        elif label=='kurtosis':
            self.speccomplexity='kurtosis'
        else:
            self.speccomplexity=label

    def run_coverage(self, event):
        """
        Controls what happens when the run coverage button is clicked

        Parameters
        ----------
        event : button press event
        """

        # First if there is already a coverage map displayed - remove this
        if self.coverage_map is not None:
            for i in range(np.size(self.coverage_map)):
                self.coverage_map[i].remove()
            self.coverage_map=None

        # calculate the coverage
        self.coverage=compute_coverage(self)
        # if covmethod is random we are going to select a random sample of
        # SAAs to retain
        if self.covmethod=='random':
            select_random_sample(self)
        # plot the coverage
        self.coverage_map=plot_coverage(self)
        # determine the total number of SAAs and spectra contained within the coverage
        self.totalsaas=get_total_saas(self)
        self.totalspec=get_total_spec(self)

        # update information window
        update_text(self.text_runcoverage, '')
        update_text(self.text_saa,'SAA size(s): '+str(self.wsaa))
        update_text(self.text_fillingfactor,'filling factor(s): '+str(self.fillfactor))
        update_text(self.text_method,'method: '+str(self.covmethod))
        update_text(self.text_spacing,'spacing: '+str(self.spacing))
        if self.refine_grid:
            if self.speccomplexity=='momdiff':
                update_text(self.text_complexity, 'complexity measure: $|m_1$-$v_p|$')
            elif self.speccomplexity=='kurtosis':
                print('')
                print(colors.fg._yellow_+"Warning: Kurtosis option is not available yet. Setting to default. "+colors._endc_)
                print('')
                #update_text(self.text_complexity, 'complexity measure: kurtosis')
                update_text(self.text_complexity, 'complexity measure: $|m_1$-$v_p|$')
            else:
                update_text(self.text_complexity, 'complexity measure: '+str(self.speccomplexity))
        else:
            update_text(self.text_complexity,'')
        update_text(self.text_totalindivsaas, 'number of SAAs to fit: '+str(self.totalsaas))
        update_text(self.text_totalindivspec, 'number of spectra to fit: '+str(self.totalspec))
        update_text(self.text_summary, 'scousepy will fit a total of '+str(str(np.sum(self.totalspec)))+' spectra.')

        # update plot
        self.fig.canvas.draw()

def get_mask(self):
    """
    Used to obtain the mask if the user provides a mask file

    """
    from astropy.io import fits
    try:
        maskfits=fits.open(self.mask_coverage)
        self._mask_found=True
        user_mask=maskfits[0].data
        user_mask[(~np.isfinite(user_mask))]=0
        user_mask[(user_mask!=0)]=1

    except IOError:
        print(colors.fg._lightred_+"File not found. Please check filepath in scousepy.config. Continuing without mask.  "+colors._endc_)
        print('')
        self._mask_found=False
        user_mask=None

    return user_mask

def compute_coverage(self):
    """
    Calculate the coverage. Sets up a grid of SAAs whose spacing is defined by
    the self.spacing parameter. The routine then checks each SAA against a mask.

    If the SAAs are chosen to have only one size, then each SAA is checked
    against the moment 0 mask.

    If multiple sizes are selected then we use a measure of spectral complexity
    in order to perform grid refinement. The mask checked against SAA is then a
    combination of the moment 0 mask and the chosen measure of spectral
    complexity.

    """
    # Get the spacing values - depends on the method and if nyquist sampled
    # or not
    if self.spacing=='nyquist':
        self.spacingvalue=[value/2. for value in self.wsaa]
    else:
        self.spacingvalue=[value for value in self.wsaa]

    # Get the locations of the SAAs
    coverage=[]
    for spacing in self.spacingvalue:
        _coverage = get_coverage(self.moments[6].shape, spacing, self.xmin, self.ymin)
        coverage.append(_coverage)

    # Now we check these locations against a mask
    if not self.refine_grid:
        # Use the moment 0 mask
        mask=self.moments[6]
        for i in range(len(self.wsaa)):
            # Remove coverage coordinates according to mask
            check_against_mask(self,coverage[i],mask,self.wsaa[i],self.fillfactor[i])
    else:
        # Get the correct measure of spectral complexity
        if self.speccomplexity=='momdiff':
            complexity_map=self.complexity_maps[0]
        elif self.speccomplexity=='kurtosis':
            complexity_map=self.complexity_maps[1]
        else:
            pass

        # get the steps
        step_values=generate_steps(complexity_map, len(self.wsaa))
        # start with the mask for the mom0
        mask=self.moments[6]
        # now modify this based on the spectral complexity
        masks=create_masks(mask, complexity_map, len(self.wsaa), step_values)

        # Now create the coverage for each mask
        for i in range(len(self.wsaa)):
            mask = masks[i]
            # Remove coverage coordinates according to mask
            all_false=check_against_mask(self,coverage[i],mask,self.wsaa[i],self.fillfactor[i])
            # if all false combine the current mask with the next one
            if (all_false) and (i != np.max(range(len(self.wsaa)))):
                masks=combine_masks(i, masks)

    return coverage

def get_coverage(shape, spacing, xmin, ymin):
    """
    Returns the central locations of SAAs

    Parameters
    ----------
    shape : ndarray
        shape of the map
    spacing : float
        spacing between the centres of SAAs

    """

    # Get the indices of the cols and rows in the momzero map where there is
    # data
    y, x = np.arange(0,shape[0]),np.arange(0,shape[1])

    # This sets the maximum extent of the coverage
    rangex = [np.min(x), np.max(x)]
    sizex = np.abs(np.min(x)-np.max(x))
    rangey = [np.min(y), np.max(y)]
    sizey = np.abs(np.min(y)-np.max(y))

    # Here we define the total number of positions in x and y for the coverage
    nposx = int((sizex/(spacing))+1.0)
    nposy = int((sizey/(spacing))+1.0)

    # This defines the coverage coordinates
    cov_x = (np.max(rangex)-(spacing)*np.arange(nposx))
    cov_y = (np.min(rangey)+(spacing)*np.arange(nposy))

    # create a grid
    cov_xx,cov_yy=np.meshgrid(cov_x,cov_y)
    cov_xx=np.flip(cov_xx,axis=1)
    # include a boolean array to be modified according to conditions
    coverage_include=np.zeros(len(cov_xx.ravel()), dtype='bool')
    # bundle everything together
    coverage=np.vstack((cov_xx.ravel(), cov_yy.ravel(),coverage_include)).T

    return coverage

def check_against_mask(self, coverage, mask, wsaa, fillfactor):
    """
    Check an SAA against a mask - used to establish which SAAs should be
    retained and which should be discarded

    Parameters
    ----------
    coverage : ndarray
        the coverage array
    mask : ndarray
        a boolean mask array
    wsaa : number
        the width of the SAAs
    fillfactor : number
        The filling fraction of the SAA. Refers to the ratio between the maximum
        number of pixels in an SAA and the number of unmasked pixels contained
        within the SAA. If this ratio is higher than the provided fillfactor the
        SAA will be retained

    """
    for i in range(len(coverage[:,0])):
        # create a local mask centred on the coverage coordinate
        localmask=mask_img(mask, centre=(coverage[i,1],coverage[i,0]), width=(wsaa,wsaa))

        # This is the maximum number of pixels contained within the area
        # defined by wsaa
        maxpix=np.sum(localmask)
        # combine the two masks
        combinedmask=localmask*mask
        # this is the total number of significant pixels within the area
        sigpix=np.sum(combinedmask)

        if maxpix == 0:
            frac=0.0
        else:
            # get the fraction
            frac=float(sigpix)/float(maxpix)

        if frac >= fillfactor:
            coverage[i,2]=True

    if not any(coverage[:,2]):
        all_false = True
    else:
        all_false = False

    return all_false

def mask_img(map, centre=None, width=None):
    """
    Accepts an image to be masked. Additionally accepts a centre location and a
    width to produce the mask. Returns a masked image

    Parameters
    ----------
    map : numpy array
        image to be masked
    centre : numpy array (optional)
        numpy array containing centre coordinates in pixel units.
    width : float (optional)
        width of the square mask

    """
    maptomask=np.ones_like(map)
    y = int(np.shape(maptomask)[0])
    x = int(np.shape(maptomask)[1])

    if centre is None: # use the middle of the image
        centre = [y//2, x//2]
    if width is None:
        width = min(centre[0], centre[1], x-centre[1], y-centre[0])

    Y, X = np.ogrid[:y, :x]
    dist_from_centre = np.zeros([y,x])

    if np.size(width)==2:
        xp=int(centre[1]+width[1]//2); xn=int(centre[1]-width[1]//2)
        yp=int(centre[0]+width[0]//2); yn=int(centre[0]-width[0]//2)
    else:
        xp=int(centre[1]+width//2); xn=int(centre[1]-width//2)
        yp=int(centre[0]+width//2); yn=int(centre[0]-width//2)

    if xn < 0:
        xn = 0
    if xp > x:
        xp = x
    if yn < 0:
        yn = 0
    if yp > y:
        yp = y

    dist_from_centre[yn:yp+1,xn:xp+1]=1.0
    mask = (dist_from_centre == 1)
    newimg = np.where(mask==1, maptomask, 0)

    return newimg

def plot_coverage(self):
    """
    Plot the coverage. Here we create patches for each of the SAAs, then to
    speed things up a bit, create a compound path from the individual SAAs for
    plotting.

    """
    import matplotlib.pylab as pl
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.path as path

    # There are potentially multiple coverages if more than one wsaa has been
    # selected, so we will store  the patches in a list
    _coverage_map=[]

    # define the colours of the coverage patches
    n = np.size(self.wsaa)
    _colors = ['dodgerblue','indianred','springgreen','yellow','magenta','cyan']

    # Cycle through each coverage
    # sortedids=sorted(range(np.size(self.wsaa)), key=lambda k: self.wsaa[k], reverse=True)
    # print(sortedids)
    for i in range(len(self.wsaa)):
        saas=[]
        coverage=self.coverage[i]
        c=_colors[i]
        w=self.wsaa[i]

        # identify if any of the SAAs are to be plotted
        if any(coverage[:,2]):
            for j in range(len(coverage[:,0])):
                if coverage[j,2]:
                    # if the SAA is to be plotted - identify the bottom left
                    # hand corner for the rectangular patch
                    bl=(coverage[j,0]-w/2., coverage[j,1]-w/2.)
                    _patch=patches.Rectangle(bl,w,w)
                    verts=_patch.get_verts()
                    saas.append(_patch)

            # get all the vertices and create a compound path - this will be
            # plotted rather than the individual SAAs as it speeds things up a
            # bit
            all_verts = np.asarray([saa.get_verts() for saa in saas])
            mypath = path.Path.make_compound_path_from_polys(all_verts)
            self.coverage_path.append(mypath)
            # create the patch
            saapatch = patches.PathPatch(mypath,alpha=0.4, facecolor=c, edgecolor='black')

            # add this to the list and plot it
            _coverage_map.append(self.map_window.add_patch(saapatch))

    return _coverage_map

def get_total_saas(self):
    """
    Method used to identify the total number of SAAs that need to be fitted
    """
    total=[]
    for i in range(len(self.wsaa)):
        coverage=self.coverage[i]
        total.append(np.sum(coverage[:,2]))
    return total

def get_total_spec(self):
    """
    Method used to identify the total number of spectra contained within each
    coverage selection
    """
    total=[]
    # get the mask
    mask=self.moments[6]
    idy,idx=np.where(mask)
    # identify all the unmasked points
    points=np.vstack((idx,idy)).T

    # Now work out which of these points are located within the coverage map
    for i in range(len(self.wsaa)):
        coverage_map=self.coverage_map[i]
        mypath=coverage_map.get_path()
        includedspectra=mypath.contains_points(points)
        total.append(np.sum(includedspectra))
    return total

def select_random_sample(self):
    """
    Method used to select a random sample of spectral averaging areas from the
    coverage. This method can be used to generate training sets.
    """
    import random
    # first get the total number of SAAs to be fit
    totalsaas=get_total_saas(self)
    for i in range(len(self.wsaa)):
        coverage=self.coverage[i]
        # identify which SAAs are to be fit
        idcov=np.where(coverage[:,2]!=0)[0]
        # get the number of SAAs to be fit
        numsaas=np.size(idcov)
        # convert this to a fraction of the total SAAs to be fit
        fractionalsample=numsaas/np.sum(totalsaas)
        # now get the number of SAAs of this size that will be included in the
        # sample
        numsample=int((fractionalsample*self.samplesize)+0.5)
        # identify which SAAs to keep and which to remove from the coverage
        _id=np.sort(random.sample(range(0,numsaas), numsample))
        idcovkeep=idcov[_id]
        idcovremove=[val for j, val in enumerate(idcov) if val not in idcovkeep]
        # remove the ones that are no longer needed
        for j in range(len(coverage[:,2])):
            if j in idcovremove:
                coverage[j,2]=0.0

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
        from astropy.visualization.wcsaxes import WCSAxes
        self._wcaxes_imported = True
    except ImportError:
        self._wcaxes_imported = False
        if self.moments[0].wcs is not None:
            warnings.warn("`WCSAxes` required for wcs coordinate display.")

    newaxis=[self.blank_window_ax[0]+0.03, self.blank_window_ax[1]+0.03, self.blank_window_ax[2]-0.06,self.blank_window_ax[3]-0.045]

    if self.scouseobject.cube.wcs is not None and self._wcaxes_imported:
        ax_image = WCSAxes(self.fig, newaxis, wcs=self.moments[0].wcs, slices=('x','y'))
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

def compute_moments(self):
    """
    Create moment maps using spectral cube
    """
    from astropy import units as u
    from scipy.stats import skew
    from scipy.stats import kurtosis

    if self.coverage_map is not None:
        for i in range(np.size(self.coverage_map)):
            self.coverage_map[i].remove()
        self.coverage_map=None
    # Trim the cube
    cube=trim_cube(self)

    # Mask the cube
    cubemask = (cube > u.Quantity(self.mask_below,cube.unit))

    # if a mask has already been input by the user then combine masks
    if self._mask_found:
        cubemask_user = np.repeat(self.user_mask[np.newaxis, :, :], cube.shape[0], axis=0)
        cubemask=cubemask.include()*cubemask_user
        cubemask=cubemask.astype('bool')

    # slab
    spectral_slab = cube.with_mask(cubemask).spectral_slab(self.velmin*u.km/u.s,self.velmax*u.km/u.s)
    # moments
    momzero = spectral_slab.moment0(axis=0)
    momone = spectral_slab.moment1(axis=0)
    momtwo = spectral_slab.linewidth_sigma()

    # momthree = spectral_slab.apply_numpy_function(skew, axis=0, nan_policy='omit',reduce=False, fill=np.nan)
    # momfour=spectral_slab.apply_numpy_function(kurtosis, axis=0, fisher=False, nan_policy='omit',reduce=False, fill=np.nan)
    # # convert to normal numpy arrays from masked arrays
    # momthree=momthree.filled(fill_value=np.nan)
    # momfour=momfour.filled(fill_value=np.nan)

    momnine = np.empty(np.shape(momone))
    momnine.fill(np.nan)

    try:
        idxmax = spectral_slab.argmax(axis=0)
    except ValueError:
        idxmax = spectral_slab.argmax(axis=0, how='ray')
    try:
        peakmap = spectral_slab.max(axis=0)
    except:
        peakmap = spectral_slab.max(axis=0, how='slice')
    bad = ~np.isfinite(peakmap) | ~np.isfinite(idxmax) | ~np.isfinite(momtwo.value)
    idxmax[bad] = 0
    momnine = spectral_slab.spectral_axis[idxmax.astype('int')].value
    momnine[bad] = np.nan
    momnine = momnine * u.km/u.s

    mask=np.zeros_like(momzero.value)
    mask[~np.isnan(momzero.value)]=1

    #moments=[momzero, momone, momtwo, momthree, momfour, momnine, mask]
    moments=[momzero, momone, momtwo, momzero, momzero, momnine, mask]
    return moments

def trim_cube(self):
    """
    Trims the x,y values of the cube, returns a trimmed cube
    """
    cube = self.scouseobject.cube[:,self.ymin:self.ymax,self.xmin:self.xmax]
    return cube

def compute_spectral_complexity(self):
    """
    This method computes different measures of spectral intensity. This can be
    updated with more functions.
    """
    momdiff_map=compute_momdiff(self.moments[1],self.moments[5])
    #kurtosis_map=self.moments[4]
    kurtosis_map=momdiff_map#self.moments[4]
    return [momdiff_map, kurtosis_map]

def compute_momdiff(mom1,vcent):
    """
    Calculate the difference between the moment one and the velocity of the
    channel containing the peak flux

    Parameters
    ----------
    momone : ndarray
        moment one (intensity-weighted average velocity) map
    vcent : ndarray
        map containing the velocities of channels containing the peak flux at
        each location

    """
    # Generate an empty array
    momdiff_map = np.empty(np.shape(mom1.value))
    momdiff_map.fill(np.nan)
    momdiff_map = np.abs(mom1.value-vcent.value)

    return momdiff_map

def generate_steps(map, nsteps):
    """
    Creates logarithmically spaced values

    Parameters
    ----------
    map : ndarray
        map of the spectral complexity measure
    nsteps : number
        number of steps of refinement
    """
    median = np.nanmedian(map)
    step_values = np.logspace(np.log10(median), \
                              np.log10(np.nanmax(map)), \
                              nsteps )

    step_values = list(step_values)
    step_values.insert(0, 0.0)
    return step_values

def create_masks(mommask, map, nmasks, step_values):
    """
    modifies the moment 0 mask according to the spectral complexity measure and
    returns a list of masks

    Parameters
    ----------
    mommask : ndarray
        moment 0 mask
    map : ndarray
        map of the spectral complexity measure
    nmasks : number
        this is the number of masks that will be created
    step_values : ndarray
        an array containing the step values
    """
    masks = []
    import matplotlib.pyplot as plt
    for i in range(nmasks):
        # create an empty mask
        mask=np.zeros_like(mommask, dtype='bool')
        # identify where map sits between the ranges in step values
        minval = step_values[i]
        maxval = step_values[i+1]
        mask[(map>=minval)&(map<=maxval)]=1
        # Now modify the moment mask
        mask=mommask*mask
        # add the mask to the list
        masks.append(mask)
    return masks

def combine_masks(idx, masks):
    """
    Combine multiple masks. This routine is used if all of the SAAs in one
    coverage map are discarded - otherwise there can be significant holes in the
    map

    Parameters
    ----------
    idx : number
        index of the current mask
    masks : ndarray
        an array containing the masks
    """
    masks[idx+1]=masks[idx+1]+masks[idx]
    masks[idx+1][np.where(masks[idx+1]!=0)]=1
    return masks

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
        empty=np.empty(np.shape(self.moments[self.moment]))
        empty[:]=np.nan
        self.map.set_data(empty)
        return self.map_window.imshow(map, origin='lower', interpolation='nearest',cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
    else:
        return self.map_window.imshow(map, origin='lower', interpolation='nearest',cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)

def append_keywords(config_file, dct, description=True):
    for key in dct.keys():
        if description:
            config_file.append(
                            '\n\n# {}'.format(dct[key]['description']))
        config_file.append('\n{} = {}'.format(key, dct[key]['default']))
    return config_file

def make_string(st):
    newstring="\'" + str(st) + "\'"
    return newstring

def make_config_file(self, description=True):
    """
    Creates an astropy table containing important coverage information
    """
    from collections import OrderedDict

    config_file = str('# ScousePy config file\n\n')

    default = [
        ('nrefine', {
            'default': np.size(self.wsaa),
            'description': "number of refinement steps"}),
        ('mask_below', {
            'default': self.mask_below,
            'description': "mask data below this value"}),
        ('mask_coverage', {
            'default': make_string(self.mask_coverage),
            'description': "optional input filepath to a fits file containing a mask used to define the coverage"}),
        ('x_range', {
            'default': [self.xmin, self.xmax],
            'description': "data x range in pixels"}),
        ('y_range', {
            'default': [self.ymin, self.ymax],
            'description': "data y range in pixels"}),
        ('vel_range', {
            'default': [self.velmin, self.velmax],
            'description': "data velocity range in cube units"}),
        ('wsaa', {
            'default': self.wsaa,
            'description': "width of the spectral averaging areas"}),
        ('fillfactor', {
            'default': self.fillfactor,
            'description': "fractional limit below which SAAs are rejected"}),
        ('samplesize', {
            'default': self.samplesize,
            'description': "sample size for randomly selecting SAAs"}),
        ('covmethod', {
            'default': make_string(self.covmethod),
            'description': "method used to define the coverage [regular/random]"}),
        ('spacing', {
            'default': make_string(self.spacing),
            'description': "method setting spacing of SAAs [nyquist/regular]"}),
        ('speccomplexity', {
            'default': make_string(self.speccomplexity),
            'description': "method defining spectral complexity"}),
        ('totalsaas', {
            'default': self.totalsaas,
            'description': "total number of SAAs"}),
        ('totalspec', {
            'default': self.totalspec,
            'description': "total number of spectra within the coverage"}),
        ]

    dct_default = OrderedDict(default)

    config_file = []

    config_file.append('[DEFAULT]')
    config_file = append_keywords(config_file, dct_default,
                                description=description)

    return config_file

def print_information(self,xloc,yloc,str,**kwargs):
    """
    GUI setup
    """
    return self.information_window.text(xloc,yloc,str,transform=self.information_window.transAxes, **kwargs)

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
