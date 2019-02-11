***********************
Installing ``scousepy``
***********************

Requirements
~~~~~~~~~~~~

``scousepy`` requires the following packages:

* `Python <http://www.python.org>`_ 3.x

* `astropy <http://www.astropy.org/>`__ >=3.0.2
* `lmfit <http://lmfit.github.io/lmfit-py/>`_ >=0.8.0
* `matplotlib <http://matplotlib.org/>`_ >=2.2.2
* `numpy <http://www.numpy.org/>`_ >=1.14.2
* `pyspeckit <http://pyspeckit.readthedocs.io/en/latest/>`_ >=0.1.21.dev2682
* `spectral_cube <http://spectral-cube.readthedocs.io/en/latest/>`_ >=0.4.4.dev1809

Note that for interactive fitting with pyspeckit you may need to customise your
matplotlib configuration. Namely, if you're using ``scousepy`` on a Mac you will
most likely need to change your backend from 'macosx' to 'Qt5Agg' (or equiv.).
You can find some information about how to do this `here <https://matplotlib.org/users/customizing.html#customizing-matplotlib>`_.

Installation
~~~~~~~~~~~~

(Available soon - stick to developer version for now - see below)

To install the latest stable release, you can type::

    pip install scousepy

or you can download the latest tar file from
`PyPI <https://pypi.python.org/pypi/scousepy>`_ and install it using::

    python setup.py install

Developer version
~~~~~~~~~~~~~~~~~

If you want to install the latest developer version of the scousepy, you can do
so from the git repository::

    git clone https://github.com/jdhenshaw/scousepy
    cd scousepy
    python setup.py install

You may need to add the ``--user`` option to the last line if you do not have
root access.
