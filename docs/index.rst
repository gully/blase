.. blase documentation master file, created by
   sphinx-quickstart on Wed Nov 18 16:09:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

blasé
===========

The `blasé` project (pronounced "blah-say") aims to improve the astronomical echelle spectroscopy data analysis process through a hybrid data-driven/model-driven approach.  It is similar in spirit to `Starfish
<http://www.github.com/iancze/Starfish>`_, and `wobblé
<https://github.com/megbedell/wobble>`_.  The key design features of `blasé` include:


1. Built on a foundation of pre-computed synthetic stellar spectra, chosen at discrete grid points
2. Extensibile for a wide range of scientific applications
3. Handles stellar, telluric, and instrumental features simultaneously

We invite you to engage with us at our `GitHub page
<http://www.github.com/gully/blase>`_, with pull requests and discussions welcomed.

.. toctree::
  :maxdepth: 1
  :caption: Navigation

   Installation <install>
   Application Programming Interface <api>
   

.. toctree::
   :maxdepth: 2

   Tutorials <tutorials/index>
   Paper Figures <paper_figures/index>
   Deep Dives <deep_dives/index>
   Frequently Asked Questions <FAQ/index>
   

This project can use either CPUs or GPUs. To check if you have a GPU available:

.. code-block:: python

  import torch

  torch.cuda.is_available()


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
