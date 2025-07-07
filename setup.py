"""Sets up the package."""

from pathlib import Path

from setuptools import find_packages, setup

# define a function that reads a file in this directory
read = lambda p: Path(Path(__file__).resolve().parent / p).read_text()

# set up the package
setup(
    name='pyRVtest',
    author='Marco Duarte, Lorenzo Magnolfi, Mikkel Solvsten, Christopher Sullivan, Anya Tarascina',
    author_email='chris.sullivan.econ@gmail.com',
    url='https://github.com/anyatarascina/pyRVtest',
    packages=find_packages(),
    python_requires='>=3.7',
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
    include_package_data=True,
    version='0.3.2'
)
