# ``scousepy``

Semi-automated multi-COmponent Universal Spectral-line fitting Engine

Copyright (c) 2017-2018 Jonathan D. Henshaw

About
=====

The ``scousepy`` package provides a method by which a large amount of complex
astronomical spectral line data can be fitted in a systematic way.

A description of the original IDL [code](https://github.com/jdhenshaw/SCOUSE)
can be found in [Henshaw et al. 2016](http://ukads.nottingham.ac.uk/abs/2016arXiv160103732H).
For a more comprehensive description of the ``scousepy`` package, including a
simple tutorial, please head over to [here](http://scousepy.readthedocs.io/en/latest/?badge=latest).

<img src="docs/source/Figure_cartoon.png"  alt="" width = "850" />

Installing ``scousepy``
=======================

Requirements
------------

``scousepy'' requires the following packages:

* [Python](http://www.python.org) 3.x

* [astropy](http://www.astropy.org/)>=3.0.2
* [lmfit](http://lmfit.github.io/lmfit-py/)>=0.8.0
* [matplotlib](https://matplotlib.org/)>=2.2.2
* [numpy](http://www.numpy.org/)>=1.14.2
* [pyspeckit](http://pyspeckit.readthedocs.io/en/latest/)>=0.1.21.dev2682
* [spectral_cube](http://spectral-cube.readthedocs.io/en/latest/)>=0.4.4.dev1809

Note that for interactive fitting with pyspeckit you may need to customise your
matplotlib configuration. Namely, if you're using ``scousepy'' on a Mac you will
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

Please help to improve this package by reporting issues via [GitHub]
(https://github.com/jdhenshaw/scousepy/issues). Alternatively, you can get in
touch at [...]

Developers
==========

This package was developed by:

* Jonathan Henshaw

[Contributors](https://github.com/jdhenshaw/scousepy/graphs/contributors) include:

* Adam Ginsburg
* Manuel Reiner

Citing ``scousepy``
===================

If you make use of this package in a publication, please consider the following
acknowledgement...

```
Henshaw et al. 2018 (in prep. coming soon)
```

Please also consider acknowledgements to the required packages in your work.

Information
===========

The method has been updated slightly from the original IDL version of the
[code](https://github.com/jdhenshaw/SCOUSE). It is now more interactive than
before which should hopefully speed things up a bit for the user. The method
is broken down into six stages in total. Each stage is summarised below.

Stage 1
-------

Here ``scousepy'' identifies the spatial area over which to fit the data. It
generates a grid of spectral averaging areas (SAAs). The user is required to
provide the width of the spectral averaging area. Extra refinement of spectral
averaging areas (i.e. for complex regions) can be controlled using the keyword
`refine_grid`.

Stage 2
-------

User-interactive fitting of the spatially averaged spectra output from stage 1.
``scousepy'' makes use of the [pyspeckit](http://pyspeckit.readthedocs.io/en/latest/)
package and is fully interactive.

Stage 3
-------

Non user-interactive fitting of individual spectra contained within all SAAs.
The user is required to input several tolerance levels to ``scousepy``. Please
refer to [Henshaw et al. 2016](http://adsabs.harvard.edu/abs/2016MNRAS.457.2675H)
for more details on each of these.

Stage 4
-------

Here SCOUSE selects the best-fits that are output in stage 3.

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
presented with several diagnostic plots namely: 'rms', 'residstd', 'redchi2',
'ncomps', 'aic', 'chi2'. These can be used to assess the quality of fits
throughout the map. Clicking on a particular region will show the spectra
associated with that location. The user can then select spectra for closer
inspection or refitting as required.

Stage 6
-------

Re-analysing the identified spectra. In this stage the user is required to
either select an alternative solution or re-fit completely the spectra
identified in stage 5.
