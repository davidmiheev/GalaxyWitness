# GalaxyWitness
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](http://choosealicense.com/licenses/mit/)
[![CodeFactor](https://www.codefactor.io/repository/github/davidosx/galaxywitness/badge/master)](https://www.codefactor.io/repository/github/davidosx/galaxywitness/overview/master)
[![Documentation Status](https://readthedocs.org/projects/galaxywitness/badge/?version=latest)](https://galaxywitness.readthedocs.io/en/latest/?badge=latest)


Package for topological data analysis of the big data. It is attempt to study distribution of galaxies in the universe via TDA. Based on [GUDHI](https://gudhi.inria.fr) and [Astropy](https://www.astropy.org)

## Requirements
1. Python 3.8+ and pip
2. git (for development)

OS X or Linux

## Installation
For the best experience, use python virtual environment:
```sh
pip3 install virtualenv
virtualenv env (or python3 -m virtualenv galaxy-witness)
. ./env/bin/activate
```
Install this package from PyPI via ```pip install galaxywitness```

## Documentation
You can find our technical documentation here: [Docs](https://galaxywitness.rtfd.io)
### For developers
[Sphinx](https://www.sphinx-doc.org/en/master/index.html) generates documentation for delelopers from sources in <code>./docs</code> folder. HTML files of documentation are in <code>./docs/build/html</code> and you can open it with browser. 
If you want to build documentation yourself:
```sh
cd docs
make html
```
or if you want to get .pdf with documentation:
```sh
cd docs
make latexpdf
```

## Usage
To run just type:
```sh   
python -m galaxywitness
```

In runtime the program will request you to choose file with your data. This file have to be in folder ```./data```

If you want to finish a work with package and deactivate virtual environment just type:
```sh
deactivate
```

## Development
You can use python virtual environment and install dependencies manually:
### Create and activate a virtual environment
This will create a new virtual environment called "galaxy-witness":
```sh
pip3 install virtualenv
virtualenv galaxy-witness (or python3 -m virtualenv galaxy-witness)
. ./galaxy-witness/bin/activate
``` 
This will clone the repository "GalaxyWitness" on your local machine, install dependencies and install this package 'galaxywitness':
```sh
git clone https://github.com/davidosx/GalaxyWitness
cd GalaxyWitness
pip install -r requirements.txt
python setup.py install
```

Or poetry package manager:
### Poetry
```sh
git clone https://github.com/davidosx/GalaxyWitness
cd GalaxyWitness
poetry install
```

### Uninstalling
For uninstalling (include dependencies and an virtual environment):
```sh
rm -r GalaxyWitness
rm -r galaxy-witness
```


