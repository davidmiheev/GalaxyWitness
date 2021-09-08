# GalaxyWitness
Package for topological analysis of galactic clusters with witness complex construction. Based on GUDHI and Simon Schoenenberger's witnesscomplex

## Requirements
1. Python 3.6+ and pip
2. git

Optional: Linux (for multiprocessing)

## Installation
You can use python virtual environment for the best experience
### Create and activate a virtual environment
This will create a new virtual environment called "galaxy-witness":

    $ pip install virtualenv
    $ virtualenv galaxy-witness (or python3 -m virtualenv galaxy-witness)
    $ . ./galaxy-witness/bin/activate
        
### Installing GalaxyWitness
This will clone the repository "GalaxyWitness" on your local machine, install dependencies and install this package 'galaxywitness':
 
    $ git clone https://github.com/DavidOSX/GalaxyWitness
    $ cd GalaxyWitness
    $ pip install -r requirements.txt
    $ python setup.py install
 
## Usage
To run just type:
    
    $ python -m galaxywitness

In runtime the program will request you to enter a number of processes for parallel computation. If you are using Linux, type value >= 1 (for example number of cores, <code>nproc</code>). If your machine doesn't run on Linux always type 1. 

If you want to finish a work with package and deactivate virtual environment just type:

    $ deacivate
## Uninstalling
Full uninstalling (include dependencies and an virtual environment)
 
    $ rm -r GalaxyWitness
    $ rm -r galaxy-witness
