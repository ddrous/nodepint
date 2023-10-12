from setuptools import setup

setup(
   name='nodepint',
   version='0.1.0',
   author='ddrous',
   author_email='desmond.ngueguin@gmail.com',
   packages=['nodepint', 'nodepint.tests'],
#    scripts=['bin/script1','bin/script2'],
   url='http://pypi.python.org/pypi/nodepint/',
   license='LICENSE.md',
   description='Parallel-in-tme learning and inference of Neural ODEs',
   long_description=open('README.md', encoding="utf-8").read(),
   install_requires=[
         "jax >= 0.3.4",
         "equinox >= 0.9.2",
         "datasets[jax] >= 2.0.0",     ## install this with --no-deps, and install all its dependencies mannually!
         "optax >= 0.1.1",
         "pytest",
         "seaborn",  ## The last two are optional
         # "pyvista[trame]",
   ],
)
