
import GalaxyWitness

project = 'GalaxyWitness'
copyright = '2021-2022, David Miheev'
author = 'David Miheev'
version = GalaxyWitness.__version__


extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme'
]

html_theme = "sphinx_rtd_theme"

html_static_path = ['_static']
html_logo = '_static/logo.png'

latex_engine = 'xelatex'
latex_elements = {
    'preamble': r"""
\usepackage{graphicx}
\usepackage{amssymb, amsmath, amsthm, mathrsfs, mathbbol}
\usepackage{fontspec, mathspec}
"""
    
}


