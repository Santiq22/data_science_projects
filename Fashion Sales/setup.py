# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 12:50:27 2024

@author: Santiago Collazo
@original_author: Krish Naik
"""
# ================================ Packages ================================== #
from setuptools import find_packages, setup
from typing import List
# ============================================================================ #

# ============================ Global parameters ============================= #
# The 'e' indicates editable mode and the . is the current directory
HYPEN_E_DOT='-e .'
# ============================================================================ #

# ======================== Get requiremnts function ========================== #
def get_requirements(file_path:str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    # List of requirements
    requirements = []
    with open(file_path) as file_obj:
        # Read the lines of the requirement.txt file
        requirements = file_obj.readlines()
        
        # Replace the newline character by the empty line one
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            # Remove '-e .' if it is in requirements.txt
            requirements.remove(HYPEN_E_DOT)
    
    return requirements
# ============================================================================ #

# ========================= Running setup function =========================== #
setup(
      name = 'Fashion sales project',
      version = '0.1',
      author = 'Santiago Collazo',
      packages = find_packages(),
      install_requires = get_requirements('requirements.txt')
)
# ============================================================================ #