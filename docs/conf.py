
import os
from pathlib import Path

import sys
sys.path.insert(0, os.path.abspath('..'))

# get the location of the source directory
source_path = Path(__file__).resolve().parent

# project information
project = 'pyRVtest'
copyright = '2023, Marco Duarte, Lorenzo Magnolfi, Mikkel Solvsten, Christopher Sullivan, and Anya Tarascina'
author = 'Marco Duarte, Lorenzo Magnolfi, Mikkel Solvsten, Christopher Sullivan, and Anya Tarascina'

# configure locations of other configuration files
html_static_path = ['static']
templates_path = ['templates']
exclude_patterns = ['_build', '_downloads', 'templates', 'templates', '**.ipynb_checkpoints']

# configure extensions
extensions = [
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'nbsphinx'
]
intersphinx_mapping = {
    'python': ('https://docs.python.org/3.6/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'patsy': ('https://patsy.readthedocs.io/en/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable/', None),
    'pyhdfe': ('https://pyhdfe.readthedocs.io/en/stable/', None),
}

# configure HTML information
html_theme = 'sphinx_rtd_theme'
