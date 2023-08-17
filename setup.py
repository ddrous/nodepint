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
      #  "pytest",
      #  "equinox",
      #  "datasets[jax]"  WARNING: They require and old version of JAX <=0.3.5. We can avoid this by installing without dependencies 'pip install datasets[jax] --no-deps', and mannully installing the dependencies one after the other.
      #  "pyvista[imageio-ffmpeg]",   WARNING: The below are for visualisation purpose only
      #  "pyvista[trame]",
      #  "matplotlib>=3.4.0",
   ],
)
