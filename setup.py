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
    python_requires='>=3.6',
    install_requires=read('requirements.txt').splitlines(),
    extras_require={
        'docs': [
            'sphinx==2.0.0', 'pandas', 'ipython', 'matplotlib', 'astunparse', 'sphinx-rtd-theme==0.4.3',
            'nbsphinx==0.5.0'
        ],
    },
    license='MIT',
    description='Code to perform econometric test of firm conduct',
    long_description=read('README.rst', 'r').strip(),
    include_package_data=True,
    version='0.1.2'
)
