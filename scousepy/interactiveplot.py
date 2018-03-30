# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

from matplotlib import pyplot
import numpy as np

class InteractivePlot:
    """
    This creates an inteactive plot of multiple spectra - the user is able to
    select spectra for which they would like to take another look at, i.e. in
    the event of a bad fit.

    Useage -
    On click - indicates the spectrum is to be checked more closely (stage 6)
    On 'r', 'd', 'esc', 'backspace' - removes selection
    On 'enter' - scousepy continues and plots the next block of spectra

    All credit for the interactive plotting goes to this:

    https://gist.github.com/smathot/2011427
    """

    def __init__(self, fig=None, ax=None, keep=False, blockrange=None,
                 blocknum_ind=0, callback=None, callback_check_spec=None):
        """
        """

        self.fig = fig
        self.ax = ax
        self.subplots = []
        self.sps = []
        self.nsubplots = np.size(self.ax)
        self.dragFrom = None
        self.keep = keep
        self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.fig.canvas.mpl_connect('button_release_event', self.release)
        self.fig.canvas.mpl_connect('scroll_event', self.scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.keyentry)
        self.done = False
        self.blockrange = blockrange
        self.blocknum_ind = blocknum_ind
        self.callback = callback
        self.callback_check_spec = callback_check_spec

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.click)
        self.fig.canvas.mpl_disconnect(self.release)
        self.fig.canvas.mpl_disconnect(self.scroll)
        self.fig.canvas.mpl_disconnect(self.keyentry)

    def show(self):
        """
        Generates the plot
        """
        if self.keep:
            pyplot.suptitle("Click: select spectrum; 'r' or 'd': Deselect spectrum; Enter: Continue or manual refit")
        else:
            pyplot.suptitle("Click: select spectrum; 'r' or 'd': Deselect spectrum; Enter: Continue")
        pyplot.tight_layout(rect=[0, 0.03, 1, 0.95])
        pyplot.show()
        return self

    def getSubPlotNr(self, event):
        """
        Returns the subplot number after you click on it
        """
        i = 0
        axisNr = None

        if np.size(self.ax) != 1:
            for axis in self.ax:
                if axis == event.inaxes:
                    axisNr = i
                    break
                i += 1
        else:
            if self.ax == event.inaxes:
                axisNr = i
        return axisNr

    def selectSubPlot(self, i):
        """
        Returns the subplot axis
        """
        return self.ax[i]

    def click(self, event):
        """
        What happens following mouse click
        """

        if self.keep:
            col='green'
        else:
            col='red'
        # First get the subplot number that was just clicked on
        subPlotNr = self.getSubPlotNr(event)
        # Check to see whether this is already in the list (we don't want
        # multiple entries)
        inlist = subPlotNr in self.subplots
        # if not then add it to the list
        if (not inlist) and (subPlotNr is not None):
            self.sps.append(subPlotNr)
        if subPlotNr is None:
            return

        if event.button == 1:
            if not inlist:
                # Draw a marker to show that the user has selected a plot
                if np.size(self.ax) == 1:
                    subPlot = self.ax
                else:
                    subPlot = self.ax[subPlotNr]
                subPlot.patch.set_facecolor(col)
                subPlot.patch.set_alpha(0.1)
                self.fig.canvas.draw()
                self.subplots = self.sps
            else:
                self.fig.canvas.draw()
        else:
            # Start a dragFrom
            self.dragFrom = event.xdata

    def keyentry(self, event):
        """
        What happens following a key entry
        """

        if (event.key == 'r') or \
           (event.key == 'backspace') or \
           (event.key == 'd') or \
           (event.key == 'escape'):

            # If any of the above are entered/selected then remove the marker
            # and the subplot from the list of plots
            subPlotNr = self.getSubPlotNr(event)
            inlist = subPlotNr in self.subplots
            if inlist:
                if np.size(self.ax) == 1:
                    subPlot = self.ax
                else:
                    subPlot = self.ax[subPlotNr]

                subPlot.patch.set_facecolor('white')
                subPlot.patch.set_alpha(0.0)

                self.sps.remove(subPlotNr)

            if subPlotNr is None:
                return

            self.fig.canvas.draw()
            self.subplots = self.sps

            return

        if event.key == 'enter':
            if pyplot.matplotlib.rcParams['interactive']:

                self.callback_check_spec(self.blockrange[self.blocknum_ind],
                                         self)

                self.sps = []
                self.subplots = []

                success = False
                while not success:
                    self.blocknum_ind += 1
                    if self.blocknum_ind < len(self.blockrange):
                        blocknum = self.blockrange[self.blocknum_ind]

                        success = self.callback(blocknum, self)
                    else:
                        success = True
                        self.done = True
                        self.disconnect()
            else:
                raise ValueError("Please use interactive mode")
                pyplot.close(self.fig.number)

                self.done = True

    def release(self, event):
        """
        This allows the user to zoom
        """
        if self.dragFrom is None or event.button != 3:
            return
        dragTo = event.xdata
        dx = self.dragFrom - dragTo
        for i in range(self.nsubplots):
            subPlot = self.selectSubPlot(i)
            xmin, xmax = subPlot.get_xlim()
            xmin += dx
            xmax += dx
            subPlot.set_xlim(xmin, xmax)
        event.canvas.draw()

    def scroll(self, event):
        """
        This allows the user to scroll
        """
        for i in range(self.nsubplots):
            subPlot = self.selectSubPlot(i)
            xmin, xmax = subPlot.get_xlim()
            dx = xmax - xmin
            cx = (xmax+xmin)/2
            if event.button == 'down':
                dx *= 1.1
            else:
                dx /= 1.1
            _xmin = cx - dx/2
            _xmax = cx + dx/2
            subPlot.set_xlim(_xmin, _xmax)
        event.canvas.draw()

def showplot(fig=None, ax=None, keep=False, blockrange=None, blocknum_ind=1,
             callback=None, callback_check_spec=None):
    """
    This is what begins the interactive plotting
    """

    pl = InteractivePlot(fig, ax, keep=keep, blockrange=blockrange,
                         blocknum_ind=blocknum_ind, callback=callback,
                         callback_check_spec=callback_check_spec)
    # first plot requires a callback
    # we need this loop because we skip any plots with no valid pixels
    success = False
    while not success:
        pl.blocknum_ind += 1
        if pl.blocknum_ind < len(pl.blockrange):
            blocknum = pl.blockrange[pl.blocknum_ind]

            success = pl.callback(blocknum, pl)
        else:
            success = True
            pl.done = True

    pl.show()
    return pl
