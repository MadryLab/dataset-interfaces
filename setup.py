from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name='dataset_interfaces',
    version='0.1.0',    
    description='Dataset Interfaces',
    url='https://github.com/MadryLab/dataset-interfaces',
    author='MadryLab',
    author_email='jvendrow@mit.edu',
    license = 'MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['torch',
                      'numpy',  
                      'torchvision',
                      'diffusers',
                      'transformers',
                      'accelerate'
                      ],
)
