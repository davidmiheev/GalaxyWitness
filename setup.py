import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='galaxywitness',
    version='0.1.6',
    author='David Miheev',
    author_email='davidhomemail29@gmail.com',
    packages=['galaxywitness'],
    url='https://github.com/DavidOSX/GalaxyWitness',
    description='Package for topological analysis of galactic clusters with witness complex construction.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
)
