.. _stats:

**********************************************************
How to output the solutions and retrieve some useful stats
**********************************************************

Below describes some of the data structures output from ``scousepy`` and how to
access some potentially important information, starting with the things that
are probably most useful for most people (with some more involved stuff further
down that can be safely ignored unless you really want to dig).

How to output the best-fitting solutions to an ascii table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assuming you have completed all four stages of the fitting process, you can
output the best-fitting solutions to an ascii table using the following commands ::

  # import scousepy
  from scousepy import scouse

  # create pointers to input, output, data
  datadir='/path/to/the/data/'
  outputdir='/path/to/output/information/'
  filename='n2h+10_37'

  # run scousepy
  config_file=scouse.run_setup(filename, datadirectory, outputdir=outputdir)
  s = scouse.stage_1(config=config_file, interactive=True)
  s = scouse.stage_2(config=config_file)
  s = scouse.stage_3(config=config_file)
  s = scouse.stage_4(config=config_file, bitesize=False)

  # output the ascii table
  from scousepy.io import output_ascii_indiv
  output_ascii_indiv(s, outputdir)

Note that in the above case, running the first four stages will simply load the
relevant information which is stored in pickle files. I have set the keyword
``bitesize=False`` here to prevent ``scousepy`` from opening the stage 4 GUI.
The final two lines will output the information to an ascii file.

Each row in the table corresponds to a single component. So any given pixel may
have multiple rows in the table. The columns correspond to

1. number of components at that pixel location
2. x location (pixels)
3. y location (pixels)
4. amplitude
5. amplitude uncertainty
6. shift (centroid velocity)
7. shift uncertainty
8. width (dispersion *not* FWHM)
9. width uncertainty
10. rms
11. standard deviation of the residual spectrum
12. :math:`\chi^{2}`
13. number of degrees of freedom
14. reduced :math:`\chi^{2}`
15. AIC

It is worth noting that a similar table will be output during stage 2. This
describes the solutions to the spectra extracted from the SAAs.

How to retrieve some useful statistics regarding the fits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``scousepy.stats`` module offers some basic but informative statistics
on the decomposition. We can compute the statistics using ::

  # import scousepy
  from scousepy import scouse

  # create pointers to input, output, data
  datadir='/path/to/the/data/'
  outputdir='/path/to/output/information/'
  filename='n2h+10_37'

  # run scousepy
  config_file=scouse.run_setup(filename, datadirectory, outputdir=outputdir)
  s = scouse.stage_1(config=config_file, interactive=True)
  s = scouse.stage_2(config=config_file)
  s = scouse.stage_3(config=config_file)
  s = scouse.stage_4(config=config_file, bitesize=False)

  # compute the statistics
  stats=scouse.compute_stats(s)

The following information can then be accessed via the ``stats`` object ::

  print("The total number of spectra in the cube is: ", stats.nspec)
  print("The total number of spectral averaging areas is: ", stats.nsaa)
  print("The total number of spectra contained within the SAA coverage is: ", stats.nspecsaa)
  print("The total number of spectra with model solutions is: ", stats.nfits)
  print("The total number of fitted components is: ", stats.ncomps)
  print("The number of components per fit is: ", stats.ncompsperfit )
  print("The total number of fits requiring multi-component models is: ", stats.nmultiple)

In addition, the user can obtain the distributions of some of the common parameters
and goodness-of-fit statistics using ::

  param_dict = stats.stats
  saa_param_dict = stats.saastats

These will return dictionaries outlining the distribution of quantities such as
the model parameters. Each statistic has an associated list that includes

1. minimum value
2. 25th percentile
3. median
4. 75th percentile
5. maximum value
6. mean value

In addition, ``stats.saastats`` includes some useful information about the SAA
fitting. It gives the distribution of the `SNR` and `alpha` parameters, as well
as the number of spectra that were fit manually (i.e. not using derivative
spectroscopy).

(Advanced) How to work your way around the SAAs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SAAs and their associated models are housed within a dictionary called ``saa_dict``.
It is indexed according to the number of levels of refinement in the SAA fitting.
For most use cases, you may only use a single SAA size, and so the dictionary
containing the SAAs can be accessed with ``mySAAdict=s.saa_dict[0]``.

Each SAA is represented as an object in this dictionary. Note that there is a SAA
for every single SAA, even if that SAA was not fit (i.e. if it did not satisfy
the masking criteria in stage 1). To get a list of SAAs that were fit you can
use something like ::

  mySAAdict=s.saa_dict[0]
  myFittedSAAs=[mySAAdict[key] for key in mySAAdict.keys() if mySAAdict[key].to_be_fit]

Each SAA object contains information such as its pixel coordinates (`saa.coordinates`),
the indices of the individual spectra included in its boundary (`saa.indices`),
the spatially averaged spectrum extracted from those pixels (`saa.spectrum`),
the measured rms (`saa.rms`), and the best-fitting model solution (`saa.model`) ::

  for saa in myFittedSAAs:
    print(saa.coordinates)

(Advanced) How to work your way around the individual spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to the SAAs, ``scousepy`` stores information on individual spectra in a
dictionary called ``indiv_dict``. The individual spectra objects contain a bunch
of information that may be useful including the pixel coordinates (`spec.coordinates`),
the index of that spectrum (`spec.index`), the measured rms (`spec.rms`). It
also contains information regarding the parent SAA. The correct index for the
SAA dictionary can be accessed (`spec.saa_dict_index`) and then the index of the
parent saa can be found (`spec.saaindex`), such that the parent SAA can be accessed
(``s.saa_dict[spec.saa_dict_index][spec.saaindex]``).

Further information regarding the fitting process can also be accessed here. The
input guesses from the parent SAA spectrum can be accessed via `spec.guesses_from_parent`.
If some of these guesses yield results that are incompatible with the tolerance
levels supplied during stage 3 then ``scousepy`` will modify the input guesses.
These can be accessed via `spec.guesses_updated`.

Finally, the models from the parent SAA(s) can be found in `spec.models_from_parent`,
and the best-fitting model is located in `spec.model` ::

  indiv_dict=s.indiv_dict
  for key, spec in indiv_dict.items():
    print(spec.index, spec.coordinates)
    print(spec.model)

(Advanced) A description of the ``scousepy`` model object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each decomposed spectrum, SAA or individual, has an associated model. The model
object contains lots of useful information regarding the model solution. The
attributes include ::

  fittype : string
      Model used during fitting (e.g. Gaussian)
  parnames : list
      A list containing the parameter names in the model (corresponds to those
      used in pyspeckit)
  ncomps : Number
      Number of components in the model solution
  params : list
      The parameter estimates
  errors : list
      The uncertainties on each measured parameter
  rms : Number
      The measured rms value
  residstd : Number
      The standard deviation of the residuals
  chisq : Number
      The chi squared value
  dof : Number
      The number of degrees of freedom
  redchisq : Number
      The reduced chi squared value
  AIC : Number
      The akaike information criterion
  fitconverge : bool
      Indicates whether or not the fit has converged
  method : String
    The method of decomposition (parent, dspec, manual, removed)
