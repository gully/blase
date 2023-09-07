# blasé

Interpretable Machine Learning for high-resolution astronomical spectroscopy.

<a href="https://blase.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Read-the%20docs-blue"></a>
<a href="https://ui.adsabs.harvard.edu/abs/2022ApJ...941..200G/abstract"><img src="https://img.shields.io/badge/Paper-Gully--Santiago & Morley (2022)-green"></a>

## _Handles stellar and telluric lines simultaneously_

We can combine stellar, [telluric](https://en.wikipedia.org/wiki/Telluric_contamination), and instrumental models into a unified forward model of your entire high-bandwidth, high-resolution spectrum. We can obtain best-in-class models of Earth's atmosphere, line-by-line, automatically, for free (or cheap).

## _Massively scalable_

By using autodiff, we can fit over 10,000 spectral lines simultaneously. This enormous amount of flexibility is unavailable in conventional frameworks that do not have [autodiff](https://en.wikipedia.org/wiki/Automatic_differentiation).  
![optimize lines](https://user-images.githubusercontent.com/860227/266395385-1d938fda-fb89-4bc7-bd88-d26a5762b8a5.gif)
^ We do this for 10,000 lines simultaneously.

## _Rooted in physics_

We first clone a precomputed synthetic spectrum, such as PHOENIX, and then **transfer learn** with data. By regularizing to the cloned model, we get the best of both worlds: data driven when the Signal-to-Noise ratio is high, and model-driven when we lack data to say otherwise.

## _Blazing fast with GPUs_

We achieve $>60 \times$ speedups with NVIDIA GPUs, so training takes minutes instead of hours.

## Get started

Visit our [step-by-step tutorials](https://blase.readthedocs.io/en/latest/tutorials/index.html) or [installation](https://blase.readthedocs.io/en/latest/install.html) pages to get started. We also have [deep dives](https://blase.readthedocs.io/en/latest/deep_dives/index.html#), or you can [read the paper](https://ui.adsabs.harvard.edu/abs/2022ApJ...941..200G/abstract). Have a question or a research project in mind? Open [an Issue](https://github.com/gully/blase/issues) or [email gully](https://gully.github.io/).

Copyright 2020, 2021, 2022, 2023 The Authors
