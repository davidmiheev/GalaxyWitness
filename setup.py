import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name='galaxywitness',
    version='0.2.3',
    license='MIT',
    author='David Miheev',
    author_email='me@davidkorol.life',
    packages=['galaxywitness'],
    url='https://github.com/DavidOSX/GalaxyWitness',
    description='Package for topological data analysis of the big data. It is attempt to study distribution of galaxies in the universe via TDA',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
    include_package_data=True,
)
