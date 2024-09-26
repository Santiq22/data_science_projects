# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 12:50:27 2024

@author: Santiago Collazo
@original_author: Krish Naik
"""
from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
      name = 'Fashion sales project',
      version = '0.1',
      author = 'Santiago Collazo',
      packages = find_packages(),
      install_requires = get_requirements('requirements.txt')
)