import ast
import copy
import datetime
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Optional, Tuple

import sys
sys.path.insert(0, os.path.abspath('..'))

import astunparse
import pyRVtest
import sphinx.application

# get the location of the source directory
source_path = Path(__file__).resolve().parent

# project information
project = 'pyRVtest'
copyright = '2023, Marco Duarte, Lorenzo Magnolfi, Mikkel Solvsten, Christopher Sullivan, and Anya Tarascina'
author = 'Marco Duarte, Lorenzo Magnolfi, Mikkel Solvsten, Christopher Sullivan, and Anya Tarascina'

# configure locations of other configuration files
html_static_path = ['static']
templates_path = ['templates']
exclude_patterns = ['_build', '_downloads', 'templates', 'notebooks', 'templates', '**.ipynb_checkpoints']

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


def clean_directories() -> None:
    """Clean directories that will be generated."""
    for name in ['_api', '_downloads', '_notebooks']:
        shutil.rmtree(source_path / name, ignore_errors=True)


def process_signature(*args: Any) -> Optional[Tuple[str, str]]:
    """Strip type hints from signatures."""
    signature = args[5]
    if signature is None:
        return None
    assert isinstance(signature, str)
    node = ast.parse(f'def f{signature}: pass').body[0]
    assert isinstance(node, ast.FunctionDef)
    node.returns = None
    if node.args.args:
        for arg in node.args.args:
            arg.annotation = None
    return astunparse.unparse(node).splitlines()[2][5:-1], ''


def setup(app: sphinx.application.Sphinx) -> None:
    """Clean directories, process notebooks, configure extra resources, and strip type hints."""
    clean_directories()
    app.connect('autodoc-process-signature', process_signature)
