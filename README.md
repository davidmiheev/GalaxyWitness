# GalaxyWitness
Package for topological analysis of galactic clusters with witness complex construction. Based on GUDHI and Simon Schoenenberger's witnesscomplex

## Requirements
1. Python 3.6+ and pip
2. git

## Installation
You can use python virtual environment for the best experience
### Create and activate a virtual environment
This will create a new virtual environment called "galaxy-witness":

    $ pip install virtualenv
    $ virtualenv galaxy-witness
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

## Uninstalling
Full uninstalling (include dependencies and an virtual environment)
 $ rm -r GalaxyWitness
 $ rm -r galaxy-witness
