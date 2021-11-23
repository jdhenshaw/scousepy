.. _tips:

***************
Tips and Tricks
***************

Bitesized Fitting
~~~~~~~~~~~~~~~~~

One thing that I have found particularly useful in my own work is to break the
fitting process up into chunks. This can be really helpful if you have a lot of
spectra to fit. I have included a bitesized fitting process into ``scousepy``
which can be run in stage 2 in the following way... ::

  if os.path.exists(datadirectory+filename+'/stage_2/s2.scousepy'):
    s.load_stage_2(datadirectory+filename+'/stage_2/s2.scousepy')
  else:
    s = scouse.stage_2(s, verbose=verb, write_ascii=True, bitesize=True, nspec=10)

  s = scouse.stage_2(s, verbose=verb, write_ascii=True, bitesize=True, nspec=100)

Check out the :ref:`Complete Example <tutorial>` in the tutorial
section of the documentation to understand what is going on here with some more
context. However, in short, the first run I have used the keywords ``bitesize=True``
and ``nspec=10``. This will fit the first 10 spectra as normal. Note the
indentation on the second call to stage 2. After the first run with 10 spectra
each subsequent call to the code will load the ``s2.scousepy`` file and then
100 spectra will be fitted until the process is complete. Of course, you can
change this to whatever value you like.
