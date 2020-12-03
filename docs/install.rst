.. _installation:

**********************************
Installing the development version
**********************************




.. note::

    Conda and pip are not available for this experimental research code.


Currently only the bleeding-edge developer version is available for beta testing.
`blase` relies on the developer version of two packages:

`torchinterp1d
<https://github.com/aliutkus/torchinterp1d>`_

and 

The HITRAN Python API `HAPI
<https://github.com/hitranonline/hapi>`_,

First, install the conda environment included in the repo::

    $ conda env create -f environment.yml
    $ conda activate blase



Then, install the two bleeding-edge packages, and finally `blase` itself ::

    $ git clone https://github.com/aliutkus/torchinterp1d.git
    $ cd torchinterp1d
    $ python setup.py develop
    $ cd ..
    $ git clone https://github.com/hitranonline/hapi.git
    $ cd hapi
    $ python setup.py develop
    $ cd ..
    $ cd blase
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
