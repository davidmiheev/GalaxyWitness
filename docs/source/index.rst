.. GalaxyWitness documentation master file, created by
   sphinx-quickstart on Fri Sep  9 20:48:37 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GalaxyWitness's documentation!
=========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

This module GalaxyWitness is written in Python and is a combination of modules `GUDHI <https://gudhi.inria.fr>`_ and `Astropy <https://www.astropy.org>`_.

The astronomical routines completely fall on the module Astropy. It is responsible for the processing of observational data and performs the transformations necessary to build a galaxy cloud to which the methods of topological data analysis are applied. And if we discard the astronomical block, then the module can be considered as a program for topological analysis of the big data. To do this, the witness complex filtration is used, which is much smaller than the classical Rips filtration on the massive point cloud (the number of galaxies on which we tested this module: :math:`\sim400000`). We also use a powerful data structure: simplex tree which allows us to efficiently store collections of simplicial complexes. 

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. * :ref:`modindex`

Contents
--------

.. toctree::
	
   description	
   witness_complex
	
