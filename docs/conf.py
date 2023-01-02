# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyRVtest'
copyright = '2023, Marco Duarte, Lorenzo Magnolfi, Mikkel Solvsten, Christopher Sullivan, and Anya Tarascina'
author = 'Marco Duarte, Lorenzo Magnolfi, Mikkel Solvsten, Christopher Sullivan, and Anya Tarascina'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

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

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
