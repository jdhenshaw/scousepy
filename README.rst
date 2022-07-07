.. image:: docs/SCOUSE_LOGO.png
   :width: 850px
   :align: center

About
=====

Multi-component spectral line decomposition with ``scousepy``. ``scousepy`` is a
package for the analysis of spectral line data. For a comprehensive description
of the algorithm and functionality please head to
`scousepy.readthedocs.io <http://scousepy.readthedocs.io/en/latest/?badge=latest>`_.

*Note*: ``scousepy`` has undergone some major updates in the latest release, namely:

* Workflow update -- scousepy now uses config files for set up. These control basic parameters for use throughout the workflow
* GUI for S1 - basic functionality is the same. Can also be run without using the config files
* GUI for S2 - added derivative spectroscopy for providing initial guesses
* Former S4 now merged into S3
* Former S5 and S6 now merged
* GUI for S3 - adaptive fit checker and fitting functionality

I am in the process of updating the documentation and tutorials. If you need
assistance on running the new version of ``scousepy``, please get in touch. For
now I have included a simple example script in the tutorials directory.

Basic description
=================

``scousepy`` includes tools for the decomposition of both data cubes, individual
spectra, and lists of specta.

Cube Fitting
------------

Cube fitting with ``scousepy`` is divided into 4 main stages::

1. Defining the coverage. Here the use informs ``scousepy`` where to fit.
``scousepy`` will compute basic noise and moments, allowing the user to define a
mask for fitting. Once defined, ``scousepy`` creates a grid of macropixels with
a user defined size and extracts a spatially averaged spectrum from each. This
can be run using the GUI or automatically using the configuration files.

2. Fitting the macropixels. ``scousepy`` uses a technique referred to as
derivative spectroscopy to identify the number of components and their key
properties. Fitting is performed via an interactive GUI.

3. Automated fitting. ``scousepy`` uses the best-fitting solutions from the
macropixels defined in stage 2 as initial guesses for an automated fitting
process that is controlled via user-defined tolerance levels.

4. Quality assessment. Here ``scousepy`` provides a GUI for quality assessment
allowing the user to visually inspect their decomposition.

Single Spectra and lists of spectra
-----------------------------------

``scousepy`` includes functionality for fitting individual or lists of spectra
using the derivative spectroscopy technique. Further information and tutorials
can be found at `scousepy.readthedocs.io <http://scousepy.readthedocs.io/en/latest/?badge=latest>`_.


Installing scousepy
===================

Requirements
------------

``scousepy`` requires the following packages:

* `Python <http://www.python.org/>`_ 3.x

* `numpy <http://www.numpy.org/>`_
* `matplotlib <https://matplotlib.org/>`_
* `astropy <http://www.astropy.org/>`_
* `lmfit <http://lmfit.github.io/lmfit-py/>`_
* `tqdm <https://github.com/tqdm/tqdm>`_
* `pyspeckit <http://pyspeckit.readthedocs.io/en/latest/>`_ >=0.1.21.dev2682
* `spectral_cube <http://spectral-cube.readthedocs.io/en/latest/>`_ >=0.4.4.dev1809

Please ensure that you are using the latest developer versions of both ``pyspeckit``
and ``spectral-cube`` (Github installation).

**Note that for interactive fitting with pyspeckit you may need to customise your
matplotlib configuration. Namely, if you're using ``scousepy`` on a Mac you will
most likely need to change your backend from 'macosx' to 'Qt5Agg' (or equiv.).
You can find some information about how to do this** `here <https://matplotlib.org/users/customizing.html#customizing-matplotlib>`_ 

Installation
------------

To install the latest version of ``scousepy``, you can type::

    git clone https://github.com/jdhenshaw/scousepy
    cd scousepy
    python setup.py install

You may need to add the ``--user`` option to the last line if you do not have
root access.

Reporting issues and getting help
=================================

Please help to improve this package by reporting issues via `GitHub <https://github.com/jdhenshaw/scousepy/issues>`_.
Alternatively, you can get in touch `here <mailto:jonathan.d.henshaw@gmail.com>`_.

Developers
==========

This package was developed by:

* Jonathan Henshaw

`Contributors <https://github.com/jdhenshaw/scousepy/graphs/contributors>`_ include:

* Adam Ginsburg
* Manuel Riener

Citing scousepy
===============

If you make use of this package in a publication, please consider the following
acknowledgements...::

  @ARTICLE{Henshaw19,
      author = {{Henshaw}, J.~D. and {Ginsburg}, A. and {Haworth}, T.~J. and
         {Longmore}, S.~N. and {Kruijssen}, J.~M.~D. and {Mills}, E.~A.~C. and
         {Sokolov}, V. and {Walker}, D.~L. and {Barnes}, A.~T. and {Contreras}, Y. and
         {Bally}, J. and {Battersby}, C. and {Beuther}, H. and {Butterfield}, N. and
         {Dale}, J.~E. and {Henning}, T. and {Jackson}, J.~M. and {Kauffmann}, J. and
         {Pillai}, T. and {Ragan}, S. and {Riener}, M. and {Zhang}, Q.},
      title = "{`The Brick' is not a brick: a comprehensive study of the structure and dynamics of the central molecular zone cloud G0.253+0.016}",
      journal = {\mnras},
      archivePrefix = "arXiv",
      eprint = {1902.02793},
      keywords = {turbulence, stars: formation, ISM: clouds, ISM: kinematics and dynamics, ISM: structure, galaxy: centre},
      year = 2019,
      month = may,
      volume = 485,
      pages = {2457-2485},
      doi = {10.1093/mnras/stz471},
      adsurl = {http://adsabs.harvard.edu/abs/2019MNRAS.485.2457H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
  }

  @ARTICLE{Henshaw2016,
         author = {{Henshaw}, J.~D. and {Longmore}, S.~N. and {Kruijssen}, J.~M.~D. and {Davies}, B. and {Bally}, J. and {Barnes}, A. and {Battersby}, C. and {Burton}, M. and {Cunningham}, M.~R. and {Dale}, J.~E. and {Ginsburg}, A. and {Immer}, K. and {Jones}, P.~A. and {Kendrew}, S. and {Mills}, E.~A.~C. and {Molinari}, S. and {Moore}, T.~J.~T. and {Ott}, J. and {Pillai}, T. and {Rathborne}, J. and {Schilke}, P. and {Schmiedeke}, A. and {Testi}, L. and {Walker}, D. and {Walsh}, A. and {Zhang}, Q.},
          title = "{Molecular gas kinematics within the central 250 pc of the Milky Way}",
        journal = {\mnras},
       keywords = {stars: formation, ISM: clouds, ISM: kinematics and dynamics, ISM: structure, Galaxy: centre, galaxies: ISM, Astrophysics - Astrophysics of Galaxies},
           year = 2016,
          month = apr,
         volume = {457},
         number = {3},
          pages = {2675-2702},
            doi = {10.1093/mnras/stw121},
  archivePrefix = {arXiv},
         eprint = {1601.03732},
   primaryClass = {astro-ph.GA},
         adsurl = {https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.2675H},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
  }

Citations courtesy of `ADS <https://ui.adsabs.harvard.edu>`__.

Please also consider acknowledgements to the required packages in your work.
