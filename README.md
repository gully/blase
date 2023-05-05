# blasé

`v0.3`

<a href="https://blase.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Read-the%20docs-blue"></a>

Hybrid data/model-driven approach to astronomical echelle spectroscopy data built on PyTorch

We do fit up to 40,000+ spectral lines simultaneously, with the magic of autodiff:
<video src='https://user-images.githubusercontent.com/860227/236579123-1acf7f23-e202-4c97-95d4-fc149ecdd191.mp4' width=680></video>

## HITRAN

The [HITRAN database](https://hitran.org/) stores atomic data used here for bespoke modeling of Earth's atmospheric absorption and emission. To tap into this database, we use the HITRAN Python API [hapi](https://hitran.org/hapi/). You may need to install the bleeding-edge version of HAPI from source.

### Authors:

- Michael Gully-Santiago (UT Austin)
- Caroline Morley (UT Austin)

Copyright 2020, 2021, 2022 Michael Gully-Santiago
