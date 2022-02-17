---
title: '`blasé`: a `PyTorch`-based modeling framework for astronomical échelle spectroscopy'
tags:
  - Python
  - astronomy
  - spectroscopy
  - stars
  - echelle
  - PyTorch
  - autodiff
authors:
  - name: Michael A. Gully-Santiago
    orcid: 0000-0002-4020-3457
    affiliation: 1
  - name: Caroline V. Morley
    orcid: 0000-0002-4404-0456
    affiliation: 1
  - name: More TBD
    affiliation: 2
affiliations:
 - name: The University of Texas at Austin Department of Astronomy, Austin, TX, USA
   index: 1
 - name: Second Affiliation goes here, Somewhere, USA
   index: 2
date: 16 February 2022
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The predictive accuracy of astronomical spectral models is exquisite but imperfect.  Individual spectral lines and groups of lines defy predictions, hampering the unbiased comparison of models to telescopic data.  Here we introduce `blasé`, a tool that learns and adapts the imperfections in astronomical spectral models.  The approach involves a two-stage training procedure that resembles transfer learning in neural networks: first training on precomputed synthetic spectral models, and then re-training on real data.  In `blasé`, the tunable model parameters possess physical significance: they are interepretable inputs of the Voigt line profile.  Tens of thousands of these lines can be adjusted to fit high-resolution échelle data across their entire available bandwidths.  The `PyTorch`-based model framework supports automatic differentiation to optimize those tens of thousands of parameters efficiently.  The framework employs sparse matrices and GPUs to achieve an unrivaled performance, with an entire training procedure taking less than a minute on modern machines.  The framework unlocks new modeling paradigms such as jointly inferring stellar and telluric lines in one fell swoop.


# Statement of need

A typical high-spectral-resolution, high-bandwidth astronomical spectrum can contain tens of thousands of spectral lines.  The line center position, amplitude, width, and shape encode information about the physical and chemical properties of the celestial object.  For many applications, it has been adequate to focus on a few information-rich lines in a spectrum to derive physical insights, with analyses proceeding by human pattern recognition.  For other more challenging applications common in exoplanet contexts, the signal-of-interest may weakly affect all or most of the spectral lines, so an automated method is desired.  We seek a general-purpose tool that can take in an astronomical spectrum and report back the line properties in one fell swoop.  Ideally this process would have some resilience to instrumental artifacts, noise, and telluric contamination.  Ideally the process would correctly account for overlapping blended spectral lines.  This general purpose tool should ideally apply to a huge range of line density and depths, and it should be fast enough to allow humans to inspect results in near-real-time.  `blasé` introduces a framework that can achieve these aims.  In essence, `blasé` creates a pixel-perfect semi-empirical model, while generating a valuable byproduct: the ledger of all of the spectral line properties.  

To our knowledge, no tool like this currently exists.  The `starfish` framework [@czekala15] supports the identification of data-model mismatches with local Gaussian Process covariance kernels.  In principle, those kernels could be catalogued, and a translation among GP hyper parameters and the intrinsic Voigt profile could be made.  But such a procedure would have to be calibrated, possibly anew for each new spectral resolution or rotation rate.  In this view, `blasé` acts like a deconvolution procedure, with the prospect of inferring super-resolution templates for high signal-to-noise-ratio (SNR) spectra.  This virtue makes `blasé` an enabling technology for Doppler Imaging, and Precision Radial Velocity (PRV) applications.

The `wobble` framework [@Bedell2019] tunes a pixel-by-pixel model of shifting stellar spectral lines in the presence of static-in-wavelength-but-dynamic-in-amplitude telluric lines.  The `wobble` model is agnostic to spectral lines, and therefore cannot infer line-by-line properties, without an additional (not-implemented) post processing step.  Hypothetically, `blasé` could operate on the outputs of `wobble`, once the stellar templates have been sanitized of telluric lines and their SNR boosted.

The `exojax` (cite XX) framework can handle tens of thousands of lines and autodifferentiate through the line properties.  But `exojax` currently makes no attempt to fit individual spectral lines independently of each other.  Only their collective changes are allowed to vary in conjunction with the bulk physical properties.

`blasé` comes closest to the existing `FAL` framework (cite XX; cite YY in prep).  These two frameworks share the same goal, but achieve it with different approaches and different assumptions.

`blasé` depends on `astropy` [@astropy13; @astropy18], `numpy` [@harris2020array], `scipy` [@scipy2020], and `PyTorch` (cite XX).

# Key innovations

Three key innovations make `blasé` transformatively performant.  First, it supports end-to-end back-propagation (cite XX), since it is built entirely in `PyTorch`.  Even the radial velocity, polynomial warping, and rotational broadening terms can sense the backpropagation.  Second, it employs sparse matrices to isolate the effect of individual spectral lines to their central sphere of influence, producing a massive speedup compared to dense models.  Third, it supports GPUs, which makes the matrix manipulations so much faster than on CPU.

# Acknowledgements

This research has made use of NASA's Astrophysics Data System Bibliographic Services.  

# References