<p align="center">
<img src="docs/SCOUSE_LOGO.png"  alt="" width = "550" />
</p>

About
=====

Multi-component spectral line decomposition with ``scousepy``. ``scousepy`` is a
package for the analysis of spectral line data. For a comprehensive description
of the algorithm and functionality please head to
[http://scousepy.readthedocs.io](http://scousepy.readthedocs.io/en/latest/?badge=latest).

Basic description
=================

``scousepy`` includes tools for the decomposition of both data cubes, individual
spectra, and lists of specta.

Cube Fitting
------------

Cube fitting with ``scousepy`` is divided into 4 main stages:

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
using the derivative spectroscopy technique.


Installing scousepy
===================

Requirements
------------

``scousepy`` requires the following packages:

* [Python](http://www.python.org) 3.x

* [astropy](http://www.astropy.org/)>=3.0.2
* [lmfit](http://lmfit.github.io/lmfit-py/)>=0.8.0
* [matplotlib](https://matplotlib.org/)>=2.2.2
* [numpy](http://www.numpy.org/)>=1.14.2
* [tqdm](https://github.com/tqdm/tqdm)
* [pyspeckit](http://pyspeckit.readthedocs.io/en/latest/)>=0.1.21.dev2682
* [spectral_cube](http://spectral-cube.readthedocs.io/en/latest/)>=0.4.4.dev1809

Please ensure that you are using the latest developer versions of both ``pyspeckit``
and ``spectral-cube`` (Github installation).

Note that for interactive fitting with pyspeckit you may need to customise your
matplotlib configuration. Namely, if you're using ``scousepy`` on a Mac you will
most likely need to change your backend from 'macosx' to 'Qt5Agg' (or equiv.).
You can find some information about how to do this [here](https://matplotlib.org/users/customizing.html#customizing-matplotlib).

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

Please help to improve this package by reporting issues via [GitHub](https://github.com/jdhenshaw/scousepy/issues).
Alternatively, you can get in touch [here](mailto:jonathan.d.henshaw@gmail.com).

Developers
==========

This package was developed by:

* Jonathan Henshaw

[Contributors](https://github.com/jdhenshaw/scousepy/graphs/contributors) include:

* Adam Ginsburg
* Manuel Riener

Citing scousepy
===============

If you make use of this package in a publication, please consider the following
acknowledgement...

```
@ARTICLE{henshaw19,
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

```

Please also consider acknowledgements to the required packages in your work.
