.. GalaxyWitness documentation master file, created by
   sphinx-quickstart on Fri Sep  9 20:48:37 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GalaxyWitness's documentation!
=========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. warning::

   This is experimental software. GalaxyWitness has only TUI. Also, the documentation is not yet complete.

GalaxyWitness is written in Python and a combination of modules `GUDHI <https://gudhi.inria.fr>`_ and `Astropy <https://www.astropy.org>`_.

The astronomical routines completely fall on the module Astropy.
It is responsible for processing observational data and performing transformations necessary to build a point cloud to which the topological data analysis methods are applied.
And if we discard the astronomical block, then the module can be considered a program for topological analysis of the big data.
To do this, the witness complex filtration is used, which is much smaller than the classical Rips filtration on the massive point cloud.
We also use a powerful data structure: a simplex tree that efficiently stores collections of simplicial complexes.

The main purpose of the program is to find the most significant topological features in the data.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. * :ref:`modindex`

Contents
--------

.. toctree::

   description
   datasets
   complexes
   clustering
