# ``scousepy``

Semi-automated multi-COmponent Universal Spectral-line fitting Engine

Copyright (c) 2017 Jonathan D. Henshaw

About
=====

The ``scousepy`` package provides a method by which a large amount of complex
astronomical spectral line data can be fitted in a systematic way. The current
version fits multiple Gaussian profiles to spectral line emission. Future
versions will include more complex models for fitting. A description of the code
can be found in [Henshaw et al. 2016](http://ukads.nottingham.ac.uk/abs/2016arXiv160103732H).
For a more comprehensive description of the ``scousepy`` package, including a
simple tutorial, please head over to:

[![Documentation Status](https://readthedocs.org/projects/scousepy/badge/?version=latest)](http://scousepy.readthedocs.io/en/latest/?badge=latest)

<img src="docs/source/Figure_cartoon.png"  alt="" width = "850" />

Installing ``scousepy``
=======================

Requirements
------------

This is what scousepy requires.

Installation
------------

This is how you install scousepy.

Developer version
-----------------

If you want to install the latest developer version of the dendrogram code, you
can do so from the git repository::

    git clone https://github.com/jdhenshaw/scousepy
    cd scousepy
    python setup.py install

You may need to add the ``--user`` option to the last line if you do not have
root access.

Reporting issues and getting help
=================================

Please help to improve this package by reporting issues via [GitHub]
(https://github.com/jdhenshaw/scousepy/issues). Alternatively, you can get in
touch at scousepy@gmail.com

Developers
==========

This package was developed by:

* Jonathan Henshaw

Citing ``scousepy``
===================

If you make use of this package in a publication, please cite the paper in which
it is presented: Henshaw et al. 2016, MNRAS, 457, 2675:

```
  @ARTICLE{2016MNRAS.457.2675H,
        author  = {{Henshaw}, J.~D. and {Longmore}, S.~N. and {Kruijssen}, J.~M.~D. and
	          {Davies}, B. and {Bally}, J. and {Barnes}, A. and {Battersby}, C. and
	          {Burton}, M. and {Cunningham}, M.~R. and {Dale}, J.~E. and {Ginsburg}, A. and
	          {Immer}, K. and {Jones}, P.~A. and {Kendrew}, S. and {Mills}, E.~A.~C. and
	          {Molinari}, S. and {Moore}, T.~J.~T. and {Ott}, J. and {Pillai}, T. and
	          {Rathborne}, J. and {Schilke}, P. and {Schmiedeke}, A. and {Testi}, L. and
                  {Walker}, D. and {Walsh}, A. and {Zhang}, Q.},
          title = "{Molecular gas kinematics within the central 250 pc of the Milky Way}",
        journal = {\mnras},
  archivePrefix = "arXiv",
         eprint = {1601.03732},
       keywords = {stars: formation, ISM: clouds, ISM: kinematics and dynamics, ISM: structure, Galaxy: centre, galaxies: ISM},
           year = 2016,
          month = apr,
         volume = 457,
          pages = {2675-2702},
            doi = {10.1093/mnras/stw121},
         adsurl = {http://ukads.nottingham.ac.uk/abs/2016MNRAS.457.2675H},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
  }
```

Please also consider adding an acknowledgment for Astropy (see
<http://www.astropy.org> for the latest recommended citation).


Information
===========

The method is broken down into seven stages. Each stage is summarised below.

Stage 1
=======
	Here SCOUSE identifies the spatial area over which to fit the data. It
	generates a grid of spectral averaging areas (SAAs). The user is required to
	provide several input values.

Stage 2
=======

	User-interactive fitting of the spatially averaged spectra output from
	stage 1.

Stage 3
=======

	Non user-interactive fitting of individual spectra contained within all SAAs.
	The user is required to input several tolerance levels to SCOUSE. Please refer
	to Henshaw+ 2016 for more details on each of these.

Stage 4
=======

	Here SCOUSE selects the best-fits that are output in stage 3.

OPTIONAL STAGES
===============

Unfortunately there is no one-size-fits-all method to selecting a best-fitting
solution when multiple choices are available (stage 4). SCOUSE uses the Akaike
Information Criterion, which weights the chisq of a best-fitting solution
according to the number of free-parameters.

While AIC does a good job of returning the best-fitting solutions, there are
areas where the best-fitting solutions can be improved. As such the following
stages are optional but *highly recommended*.

Given the level of user interaction, this is the most time-consuming part of the
routine. However, changing the tolerance levels in stage 3 can help. A quick run
through of stage 5 is recommended to see whether or not the tolerance levels
should be changed. Once the user is satisfied with the tolerance levels of
stage 3, a more detailed inspection of the spectra should take place.

Depending on the data a user may wish to perform a few iterations of Stages 5-7.

Stage 5
=======

	Checking the fits. Here the user is required to check the best-fitting
	solutions to the spectra. The user enters the indices of spectra that they
	would like to revisit. One can typically expect to re-analyse (stage 6) around
	5-10% of the fits. However, this is dependent on the complexity of the
	spectral line profiles.

Stage 6
=======

	Re-analysing the identified spectra. In this stage the user is required to
	either select an alternative solution or re-fit completely the spectra
	identified in stage 5.

Stage 7
=======

	SCOUSE then integrates these new solutions into the solution file.
