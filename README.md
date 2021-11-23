<p align="center">
<img src="docs/SCOUSE_LOGO.png"  alt="" width = "550" />
</p>

About
=====

Multi-component spectral line decomposition with ``scousepy``. ``scousepy'' is a
package for the analysis of spectral line data. For a comprehensive description
of the algorithm and functionality please head to
[http://scousepy.readthedocs.io](http://scousepy.readthedocs.io/en/latest/?badge=latest).

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
* [pyspeckit](http://pyspeckit.readthedocs.io/en/latest/)>=0.1.21.dev2682
* [spectral_cube](http://spectral-cube.readthedocs.io/en/latest/)>=0.4.4.dev1809

Note that for interactive fitting with pyspeckit you may need to customise your
matplotlib configuration. Namely, if you're using ``scousepy`` on a Mac you will
most likely need to change your backend from 'macosx' to 'Qt5Agg' (or equiv.).
You can find some information about how to do this [here](https://matplotlib.org/users/customizing.html#customizing-matplotlib).

Installation
------------

(Available soon - stick to developer version for now - see below)

To install the latest stable release, you can type::

    pip install scousepy

or you can download the latest tar file from
[PyPI](https://pypi.python.org/pypi/scousepy) and install it using::

    python setup.py install

Developer version
-----------------

If you want to install the latest developer version of the scousepy, you can do
so from the git repository::

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

Information
===========

The method has been updated slightly from the original [IDL version](https://github.com/jdhenshaw/SCOUSE)
of the code. It is now more interactive than before which should hopefully speed
things up a bit for the user. The method is broken down into six stages in
total. Each stage is summarised below.

<img src="docs/source/Figure_cartoon.png"  alt="" width = "850" />


Stage 1
-------

Here ``scousepy`` identifies the spatial area over which to fit the data. It
generates a grid of spectral averaging areas (SAAs). The user is required to
provide the width of the spectral averaging area. Extra refinement of spectral
averaging areas (i.e. for complex regions) can be controlled using the keyword
`refine_grid`.

Stage 2
-------

User-interactive fitting of the spatially averaged spectra output from stage 1.
``scousepy`` makes use of the [pyspeckit](http://pyspeckit.readthedocs.io/en/latest/)
package and is fully interactive.

Stage 3
-------

Non user-interactive fitting of individual spectra contained within all SAAs.
The user is required to input several tolerance levels to ``scousepy``. Please
refer to [Henshaw et al. 2016](http://adsabs.harvard.edu/abs/2016MNRAS.457.2675H)
for more details on each of these.

Stage 4
-------

Here ``scousepy`` selects the best-fits that are output in stage 3.

OPTIONAL STAGES
===============

Unfortunately there is no one-size-fits-all method to selecting a best-fitting
solution when multiple choices are available (stage 4). SCOUSE uses the Akaike
Information Criterion, which weights the chi-squared value of a best-fitting
solution according to the number of free-parameters.

While AIC does a good job of returning the best-fitting solutions, there are
areas where the best-fitting solutions can be improved. As such the following
stages are optional but *highly recommended*.

This part of the process has changed significantly from the original code. The
user is now presented with several diagnostic plots (see below), selecting
different regions will display the corresponding spectra, allowing the user to
check the fit quality.

Depending on the data a user may wish to perform a few iterations of Stages 5-6.

Stage 5
-------

Checking the fits. Here the user is required to check the best-fitting
solutions to the spectra. This stage is now fully interactive. The user is first
presented with several diagnostic plots namely: `rms`, `residstd`, `redchi2`,
`ncomps`, `aic`, `chi2`. These can be used to assess the quality of fits
throughout the map. Clicking on a particular region will show the spectra
associated with that location. The user can then select spectra for closer
inspection or refitting as required.

Stage 6
-------

Re-analysing the identified spectra. In this stage the user is required to
either select an alternative solution or re-fit completely the spectra
identified in stage 5.
