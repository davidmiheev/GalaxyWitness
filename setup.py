import os
import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name='GalaxyWitness',
    version='0.2.1',
    license='MIT',
    author='David Miheev',
    author_email='me@davidkorol.life',
    packages=['GalaxyWitness'],
    url='https://github.com/DavidOSX/GalaxyWitness',
    description='Package for topological analysis of galactic clusters with witness complex construction.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
    include_package_data=True,
)

print("\nBuilding documentation with \033[01;32mSphinx\033[0m...")
os.system('sphinx-build -b html docs/source/ docs/build/html')


