
project = 'GalaxyWitness'
author = 'David Miheev'


extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax'
]

html_theme = "sphinx_rtd_theme"

html_static_path = ['_static']
html_logo = '_static/logo.png'

latex_engine = 'xelatex'
latex_elements = {
    'preamble': r'''
\usepackage{amssymb, latexsym, amsmath, amsthm, mathrsfs, mathspec, mathbbol}
'''
}


