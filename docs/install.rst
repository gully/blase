.. _installation:

**********************************
Installing the development version
**********************************




.. note::

    Conda and pip are not available for this experimental research code.


First, install a conda environment included in the repo.  These are located in 
the `conda_environments/` sub-directory.  These will merely serve as a substrate 
onto which a fresh version of blasé will be installed::

    $ conda env create -f environment.yml
    $ conda activate blase
    
Then, install the bleeding-edge developer version of `blase` ::

    $ cd blase
    $ python setup.py develop


And voila!  It should work.  You can run the tests in `tests` to double-check
and benchmark GPU/CPU performance::

    $ py.test -vs



The first-principles telluric forward modeling requires the HITRAN Python API `HAPI
<https://github.com/hitranonline/hapi>`_.  This arm of the code is still highly experimental
so it is not recommended for general use.  Instead we recommend starting from a TelFit 
template, which then circumvents the need for HITRAN.  Eventually these two arms
of the code may merge.  If you want to experiment with it, you will need to install HAPI::

    $ git clone https://github.com/hitranonline/hapi.git
    $ cd hapi
    $ python setup.py develop


Requirements
============

The project may work with a variety of Python 3 minor versions, though few have been tested.  The project has been developed with:

- Python: 3.8
- PyTorch: 1.7--1.11
- CUDA 10.2+
