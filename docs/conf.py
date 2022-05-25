# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../src/"))

# -- Project information -----------------------------------------------------

project = "blase"
copyright = "2020, gully"
author = "gully"

# The full version, including alpha/beta/rc tags
release = "0.3"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx_gallery.load_style",
]

autodoc_mock_imports = ["torchinterp1d", "hapi"]
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


nbsphinx_thumbnails = {
    "tutorials/demo1": "_static/397px-PyTorch_logo_icon.svg.png",
    "tutorials/demo2": "_static/397px-PyTorch_logo_icon.svg.png",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


html_theme_options = {
    "repository_url": "https://github.com/gully/blase",
    "use_repository_button": True,
    "repository_branch": "main",
    "announcement": "🆕 blasé will be featured in a talk at the <a href="https://thibaultmerlephd.wixsite.com/cs21ml">Machine Learning Splinter Session</a> at the <a href="https://coolstars21.github.io">Cool Star 21 Conference</a> in Toulouse, France on July 5th, 2022!",
}
