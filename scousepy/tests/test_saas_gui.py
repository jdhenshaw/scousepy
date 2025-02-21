import matplotlib.pyplot as plt
import numpy as np
from spectral_cube import SpectralCube

from scousepy.core.interative.SAAsGUI import SAAsGUI
from scousepy.core.scouse import scouse


def test_gui():
    # Load a test cube
    cube = SpectralCube.read(
        "/Users/henshaw/Dropbox/Work/Code/GitHub/scousepy_tutorials/data/n2h+10_37_strip.fits"
    )

    # Create scouse object
    scouseobject = scouse()
    scouseobject.cube = cube

    # Create the GUI
    gui = SAAsGUI(scouseobject=scouseobject, verbose=True)

    # Add some test data to plot
    test_data = np.random.rand(100, 100)
    gui.map_window.window.imshow(cube[0, :, :].value, cmap=gui.cmap)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    test_gui()
