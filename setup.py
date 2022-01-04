from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='pyRVtest',
    url='https://github.com/chrissullivanecon/pyRVtest',
    author='Marco Duarte, Lorenzo Magnolfi, Mikkel Solvsten, and Christopher Sullivan',
    author_email='cjsullivan@wisc.edu',
    # Needed to actually package something
    packages=['pyRVtest'],
    # Needed for dependencies
    install_requires=['numpy', 'patsy', 'pyhdfe', 'scipy', 'sympy', 'pandas', 'statsmodels'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='An example of a python package from pre-existing code',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.txt').read(),
)
