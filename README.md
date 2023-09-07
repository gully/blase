# blasé

Interpretable Machine Learning for high-resolution astronomical spectroscopy.

<a href="https://blase.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Read-the%20docs-blue"></a>
<a href="https://ui.adsabs.harvard.edu/abs/2022ApJ...941..200G/abstract"><img src="https://img.shields.io/badge/Paper-Gully--Santiago & Morley (2022)-green"></a>

## Massively scalable

By using autodiff, we can fit over 40,000 spectral lines simultaneously.
<video width="320" autoplay loop muted playsinline>

  <source src="'https://user-images.githubusercontent.com/860227/236579123-1acf7f23-e202-4c97-95d4-fc149ecdd191.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

## Rooted in physics

We first clone a precomputed synthetic spectrum, such as PHOENIX, and then **transfer learn** with data.

## Blazing fast with GPUs

We achieve $>60 \times$ speedups with NVIDIA GPUs, so training takes minutes instead of hours.

Copyright 2020, 2021, 2022, 2023 The Authors
