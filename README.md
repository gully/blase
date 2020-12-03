# blasé

Hybrid data/model-driven approach to astronomical echelle spectroscopy data built on PyTorch

## Interpolation

Interpolation is needed in order to fit for the radial velocity of an observed spectrum compared to a spectral template. Currently PyTorch has not implemented interpolation for irregularly specified input coordinates, though the PyTorch team is aware of the interest in this [Feature Request](https://github.com/pytorch/pytorch/issues/1552).

Instead, we are using a third party solution for interpolation found here:  
https://github.com/aliutkus/torchinterp1d

So in order to run this code you must install and compile this custom package manually:

```python
git clone https://github.com/aliutkus/torchinterp1d.git
cd torchinterp1d
python setup.py develop
```

## HITRAN

The [HITRAN database](https://hitran.org/) stores atomic data used here for bespoke modeling of Earth's atmospheric absorption and emission.  To tap into this database, we use the HITRAN Python API [hapi](https://hitran.org/hapi/).  You will need to install the bleeding-edge version of HAPI from source using the same pattern as for `torchinterp1d`, but for `hapi` instead.





### Authors:

- Michael Gully-Santiago (UT Austin)

Copyright The Authors 2020
