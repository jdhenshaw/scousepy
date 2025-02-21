import matplotlib.pyplot as plt
import numpy as np

from ..utils.logger import catch_warnings, warning_to_log


class SAAsGUI(object):
    """
    A class for the interactive generate SAAs GUI.
    """

    class Window:
        def __init__(self, parent, position, color="white"):
            self.parent = parent  # reference to main GUI
            self.ax = np.array(position, dtype=float)
            self.window = None
            self.color = color

        def _setup_window(self, tick_params=False):
            """
            Setup the window.
            """
            self.window = self.parent.fig.add_axes(self.ax)
            if tick_params:
                self.window.tick_params(
                    axis="both",
                    which="both",
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelbottom=False,
                    labeltop=False,
                    labelleft=False,
                    labelright=False,
                )
            return self

        def _setup_map_window(self):
            """
            Setup the map window.
            """
            try:
                from astropy.visualization.wcsaxes import WCSAxes

                self._wcaxes_imported = True
            except ImportError:
                self._wcaxes_imported = False
            if (
                self.parent.scouseobject.cube[0, :, :].wcs is not None
                and not self._wcaxes_imported
            ):
                self.parent.logger.printlog(
                    message="`WCSAxes` required for wcs coordinate display.",
                    status="warning",
                    print_to_terminal=self.parent.verbose,
                )
            newaxis = [
                self.ax[0] + 0.03,
                self.ax[1] + 0.03,
                self.ax[2] - 0.06,
                self.ax[3] - 0.045,
            ]

            with warning_to_log():
                if (
                    self.parent.scouseobject.cube.wcs is not None
                    and self._wcaxes_imported
                ):
                    ax_image = WCSAxes(
                        self.parent.fig,
                        newaxis,
                        wcs=self.parent.scouseobject.cube[0, :, :].wcs,
                        slices=("x", "y"),
                    )
                    self.window = self.parent.fig.add_axes(ax_image)
                    x = self.window.coords[0]
                    y = self.window.coords[1]
                    x.set_axislabel("  ")
                    y.set_axislabel("  ")
                    x.set_ticklabel(exclude_overlapping=True)
                    y.set_ticklabel(
                        rotation=90,
                        verticalalignment="bottom",
                        horizontalalignment="left",
                        exclude_overlapping=True,
                    )
                else:
                    self.window = self.parent.fig.add_axes(newaxis)

            return self

    @catch_warnings
    def __init__(
        self, scouseobject: "scouse", verbose: bool = True, logger: "Logger" = None
    ):
        """
        Initialize the SAAsGUI class.

        Parameters
        ----------
        scouseobject : scouse
            The scouse object.
        verbose : bool, optional
            Whether to print verbose output.
        logger : Logger, optional
            The logger to use.
        """
        with warning_to_log():
            super().__init__()
            self.logger = logger

            self.scouseobject = scouseobject
            self.verbose = verbose
            self._setup_gui()

    def _setup_gui(self):
        """
        Setup the GUI.
        """
        self._set_rcParams()
        self.cmap = "Greys_r"
        self.fig = plt.figure(figsize=(14, 8))

        self.blank_window_ax = [0.1, 0.2, 0.375, 0.6]
        self.blank_window = self.Window(self, self.blank_window_ax)._setup_window(
            tick_params=False
        )
        self.map_window = self.Window(self, self.blank_window_ax)._setup_map_window()
        self.information_window_ax = [0.575, 0.3, 0.35, 0.4]
        self.information_window = self.Window(
            self, self.information_window_ax
        )._setup_window(tick_params=False)

    def _set_rcParams(self):
        """
        Set the rcParams.
        """
        from matplotlib import rcParams

        rcParams["font.family"] = "Arial"
        rcParams["font.size"] = 9
        rcParams["lines.linewidth"] = 1.0  ## line width in points
        rcParams["axes.labelsize"] = 10  ## fontsize of the x any y labels
        rcParams["xtick.labelsize"] = 10  ## fontsize of the tick labels
        rcParams["ytick.labelsize"] = 10  ## fontsize of the tick labels
        rcParams["xtick.major.pad"] = 4  ## distance to major tick label in points
        rcParams["ytick.major.pad"] = 4  ## distance to major tick label in points
        rcParams["xtick.major.size"] = 4  ## major tick size in points
        rcParams["xtick.minor.size"] = 2  ## minor tick size in points
        rcParams["xtick.major.width"] = 1.0  ## major tick width in points
        rcParams["xtick.minor.width"] = 1.0  ## minor tick width in points
        rcParams["ytick.major.size"] = 4  ## major tick size in points
        rcParams["ytick.minor.size"] = 2  ## minor tick size in points
        rcParams["ytick.major.width"] = 1.0  ## major tick width in points
        rcParams["ytick.minor.width"] = 1.0  ## minor tick width in points
        rcParams["xtick.direction"] = "in"
        rcParams["ytick.direction"] = "in"

        # remove some matplotlib keyboard shortcuts to prevent meltdown
        if "q" in plt.rcParams["keymap.quit"]:
            plt.rcParams["keymap.quit"].remove("q")
        if "Q" in plt.rcParams["keymap.quit_all"]:
            plt.rcParams["keymap.quit_all"].remove("Q")

        plt.ioff()
