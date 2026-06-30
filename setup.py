"""Sets up the package."""

from pathlib import Path

from setuptools import find_packages, setup

# define a function that reads a file in this directory.
# Explicit utf-8 — Windows defaults to cp1252 via Path.read_text() and
# fails on README.rst's non-ASCII characters during pip install.
read = lambda p: Path(Path(__file__).resolve().parent / p).read_text(encoding='utf-8')

# set up the package
setup(
    name='pyRVtest',
    author='Marco Duarte, Lorenzo Magnolfi, Mikkel Solvsten, Christopher Sullivan, Anya Tarascina',
    author_email='chris.sullivan.econ@gmail.com',
    url='https://github.com/anyatarascina/pyRVtest',
    packages=find_packages(),
    # CI exercises Python 3.11 only (.github/workflows/ci.yml); 3.7/3.8
    # are EOL and not tested. 3.9+ is the supported floor.
    python_requires='>=3.9',
    install_requires=read('requirements.txt').splitlines(),
    extras_require={
        'docs': [
            'sphinx==5.0.2', 'pandas', 'ipython', 'astunparse', 'sphinx-rtd-theme==1.1.1',
            'nbsphinx==0.8.11', 'jinja2==3.0.3', 'docutils==0.17.1', 'numpydoc'
        ],
    },
    license='MIT',
    description='Code to perform econometric test of firm conduct',
    long_description=read('README.rst').split('docs-start')[1].strip(),
    long_description_content_type='text/x-rst',
    include_package_data=True,
    version='0.4.0b5'
)
