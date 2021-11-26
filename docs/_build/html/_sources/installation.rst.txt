***********************
Installing ``scousepy``
***********************

Requirements
~~~~~~~~~~~~

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

Note that for interactive fitting with pyspeckit you may need to customise your
matplotlib configuration. Namely, if you're using ``scousepy`` on a Mac you will
most likely need to change your backend from 'macosx' to 'Qt5Agg' (or equiv.).
You can find some information about how to do this `here <https://matplotlib.org/users/customizing.html#customizing-matplotlib>`_

Installation
~~~~~~~~~~~~

To install the latest version of ``scousepy``, you can type::

    git clone https://github.com/jdhenshaw/scousepy
    cd scousepy
    python setup.py install

You may need to add the ``--user`` option to the last line if you do not have
root access.
