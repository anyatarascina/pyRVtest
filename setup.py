import re

from pathlib import Path

from setuptools import find_packages, setup

read = lambda p: Path(Path(__file__).resolve().parent / p).read_text()

version_match = re.search(r'^__version__ = \'([^\']*)\'', read('pyblp/version.py'), re.M)
assert version_match is not None


setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='pyRVtest',
    url='https://github.com/chrissullivanecon/pyRVtest',
    author='Marco Duarte, Lorenzo Magnolfi, Mikkel Solvsten, and Christopher Sullivan',
    author_email='chris.sullivan.econ@gmail.com',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=read('requirements.txt').splitlines(),
    license='MIT',
    description='Code to perform econometric test of firm conduct',
    long_description=open('README.md').read(),
    include_package_data=True,
    version=version_match.group(1)
)
