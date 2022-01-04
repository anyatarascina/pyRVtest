from pathlib import Path

from setuptools import find_packages, setup


# define a function that reads a file in this directory
read = lambda p: Path(Path(__file__).resolve().parent / p).read_text()
setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='pyRVtest',
    url='https://github.com/chrissullivanecon/pyRVtest',
    author='Marco Duarte, Lorenzo Magnolfi, Mikkel Solvsten, and Christopher Sullivan',
    author_email='chris.sullivan.econ@gmail.com',
    # Needed to actually package something
   packages=find_packages(),
    # Needed for dependencies
    install_requires=read('requirements.txt').splitlines(),
    # *strongly* suggested for sharing
    version='0.1.0',
    # The license can be anything you like
    license='MIT',
    description='Code to perform econometric test of firm conduct',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)
