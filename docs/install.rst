.. _installation:

**********************************
Installing the development version
**********************************




.. note::

    Conda and pip are not available for this experimental research code.


Currently only the bleeding-edge developer version is available for beta testing.
First, install the conda environment included in the repo::

    $ conda env create -f environment.yml
    $ conda activate blase
    $ python setup.py develop

And voila!  It should work.  You can run the tests in `tests` to double-check
and benchmark GPU/CPU performance::

    $ py.test -vs



Requirements
============

The project may work with a variety of Python 3 minor versions, though none have been tested.  The project has been developed with:

- Python: 3.8
- PyTorch: 1.7
- CUDA 10.2
