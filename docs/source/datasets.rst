Datasets
========
The Galaxy Witness project provides a set of datasets and a TAP service to query them.

Currently, you can use several datasets and TAP service. Soon, we plan to add more datasets containing a large corpus of data.

Available datasets
------------------
In this section, we describe the datasets available in the terminal user interface and their characteristics.

* Galaxies_400K (Galaxies_400K.csv)
* Galaxies_1KK (Galaxies_1KK.csv)

TAP service
-----------
The TAP service is a web service that allows you to query the available datasets. 
In this section, we describe the TAP services available in the tterminal user interface and their characteristics.

* RCSED ("http://rcsed-vo.sai.msu.ru/tap/")
* Simbad ("http://simbad.u-strasbg.fr/simbad/sim-tap")
* Ned ("http://ned.ipac.caltech.edu/tap")
* Coming soon

Programming interface for datasets
----------------------------------

.. autoclass:: galaxywitness.datasets.Dataset
	 :members: