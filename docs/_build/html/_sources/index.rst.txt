.. image:: ./SCOUSE_LOGO.png
  :align: center
  :width: 750

|

********
scousepy
********

Multi-component spectral line decomposition with ``scousepy``. ``scousepy`` is a
package for the analysis of spectral line data. A detailed
description of the method (and the original `IDL version
<https://github.com/jdhenshaw/SCOUSE>`_ of the code)
can be found in `Henshaw et al. 2016 <http://ukads.nottingham.ac.uk/abs/2016arXiv160103732H>`_.
In the following pages you will find a :ref:`brief introduction <description>`
to the method as well as a :ref:`tutorial <tutorial>`. The `source code
<https://github.com/jdhenshaw/scousepy>`_ is available on GitHub and comments
and contributions are very welcome.

Documentation
~~~~~~~~~~~~~

.. toctree::
  :maxdepth: 4

  installation.rst
  description.rst
  tutorial.rst
  tips.rst
  license.rst

Reporting issues and getting help
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please help to improve this package by reporting `issues
<https://github.com/jdhenshaw/scousepy/issues>`_ via GitHub. Alternatively, if
you have any questions or if you are having any problems getting set up you can
get in touch `here <mailto:jonathan.d.henshaw@gmail.com>`_.

Developers
~~~~~~~~~~

This package was developed by:

* Jonathan Henshaw

Contributors include:

* Adam Ginsburg
* Manuel Riener

Citing scousepy
~~~~~~~~~~~~~~~

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

Papers using scousepy
~~~~~~~~~~~~~~~~~~~~~

* Henshaw et al. 2021, MNRAS, accepted
* Liu et al. 2021, MNRAS, submitted
* Barnes et al. 2021, MNRAS, 503, 4601
* Callanan et al. 2021, MNRAS, 505, 4310
* Yue et al. 2021, RAA, 21, 24
* Cosentino et al. 2020, MNRAS, 499, 1666
* Henshaw et al. 2020, Nat. Ast. 4, 1064
* Henshaw et al. 2019, MNRAS, 485, 2457
* Cosentino et al. 2018, MNRAS, 474, 3760
* Henshaw, Longmore, Kruijssen, 2016, MNRAS, 463L, 122
* Henshaw et al. 2016, MNRAS, 457, 2675

Recipe
~~~~~~

Recipe for a fine Liverpudlian `Scouse pie
<http://www.bbc.co.uk/food/recipes/scouse_pie_49004>`_.
