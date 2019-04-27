.. _tutorial:

********
Tutorial
********

The aim of this tutorial is to give a basic introduction to the main workflow of
``scousepy``. Each stage will be described below along with some of the
important customisable keywords. All data for the tutorial can be found `here
<https://github.com/jdhenshaw/scousepy_tutorials>`_, along with some example
scripts.

Data
~~~~

This tutorial utilises observations of N2H+ (1-0) towards the Infrared Dark
Cloud G035.39-00.33. This data set was first published in `Henshaw et al. 2013.
<http://adsabs.harvard.edu/abs/2013MNRAS.428.3425H>`_.
These observations were carried out with the IRAM 30m Telescope. IRAM is
supported by INSU/CNRS (France), MPG (Germany) and IGN (Spain). The data file
is in fits format and is named ::

  n2h+10_37.fits

Getting Started
~~~~~~~~~~~~~~~

``scousepy`` requires several key parameters that are worth setting as global
parameters. These can be set in the following way ::

  filename =  'n2h+10_37'
  datadirectory =  './'
  wsaa =  [8.0]

  outputdir = './myoutputdirectory/' # optional
  ppv_vol = [32.0,42.0,None,None,None,None] # optional
  mask_below = 0.3 # optional

Here ``filename`` is provided without the ``.fits`` extension. ``datadirectory``
points to the location of the data. By default ``scousepy`` will create a new
directory in the same location as the input file. However, this can be changed
with the optional keyword ``outputdir``. ``wsaa`` is used during stage 1. It
defines the maximum width of the region used to generate spatially averaged
spectra (see below).

The optional keywords listed above include ``ppv_vol``, which allows the user to
control the region of the datacube over which to perform the fitting, and
``mask_below`` which is used to mask the integrated intensity map produced
during stage 1. ``ppv_vol`` is provided as a list and given in the format
``ppv_vol = [min_vel, max_vel, min_y, max_y, min_x, max_x]``.
These are the ranges over which the fitting will be performed. This tutorial
is tuned to the n2h+ data, which has hyperfine structure. Here we implement the
range 32-42 km/s since this allows us to focus on the isolated hyperfine
component, which we model as a Gaussian.

Stage 1
~~~~~~~

As discussed in the :ref:`description <description>` of the code, the purpose
of stage 1 is to identify the spatial area over which to fit the data. Stage 1
requires several key parameters. To run stage 1 you can use ::

  s = scouse.stage_1(filename, datadirectory, wsaa,
                     ppv_vol=ppv_vol,
                     outputdir=outputdir,
                     mask_below=mask_below,
                     fittype='gaussian',
                     verbose = True,
                     write_moments=True,
                     save_fig=True)

Here I've used the keywords ``write_moments=True`` and ``save_fig=True``. The
former will produce FITS files of the zeroth, first, and second order moments of
the data. It will also create a FITS file containing the spectral coordinate of
the maximum value of the spectrum. The latter keyword creates a simple plot of
the coverage that will be used during the fitting procedure. The terminal output
should look something like this...

.. image:: ./stage1.png
  :align: center
  :width: 900

Where it tells us that we will have to fit a total of 12 spatially averaged
spectra and that the total number of spectra to fit is 126. The output coverage
map for this particular tutorial is not much to look at, but here it is anyway...

.. image:: ./n2h+10_37_coverage.png
  :align: center
  :width: 300

Stage 2
~~~~~~~

Stage 2 is where we will perform our manual fitting. It is simple to run using ::

  s = scouse.stage_2(s, verbose=True, write_ascii=True)

where the keyword ``write_ascii`` has been set to output the best-fitting
solutions as an ascii file at the end of the fitting procedure. The fitting
process is based on the interactive process of `pyspeckit
<https://github.com/pyspeckit/pyspeckit>`_. Initialising the fitter will look
a bit like this..

.. image:: ./stage2_1.png
  :align: center
  :width: 900

where we will have an indication of how many spectra we have to fit (and how
many we have already fitted), as well as some important info for the ``pyspeckit``
interactive fitter. Upon running stage 2, a window should have popped up where
one of the spatially averaged spectra will be displayed. Interactive fitting
can be performed using several commands. To indicate components you would like
to fit select each component twice, once somewhere close to the peak emission
and another click to indicate (roughly) the full-width at half-maximum. In my
experience with this, you don't need to be particularly accurate, ``pyspeckit``
does an excellent job of picking up the components you have selected. Selection
can be made either using the keyboard (`m`) or mouse. Once selected this will
look something like this...

.. image:: ./stage2_2.png
  :align: center
  :width: 400

If you are happy with your fit, hitting `d` will lock it in. The resulting
fit will be plotted. At this point ``scousepy`` will output some useful information
to the terminal...

.. image:: ./stage2_3.png
  :align: center
  :width: 900

and will ask if you're happy with the fit. If the fit looks good, press enter
to continue. This will lock the fit in and overplot the individual components...

.. image:: ./stage2_4.png
  :align: center
  :width: 400

``scousepy`` will then move onto the next spectrum. If you're not happy with the
fit you can always re-enter the interactive fitter by typing `f`. Repeat this
process until the process is completed.

For large datasets its worth noting that there are a couple of keywords here
that might be useful, particularly ``bitesize``. This enables the user to
perform bitesize fitting where the process is broken down into sessions and the
user fits a fixed number of spectra at any one time. The number of spectra to
fit in any one session can be controlled using the ``nspec`` keyword.
