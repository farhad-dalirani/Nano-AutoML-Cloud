from setuptools import find_packages, setup
from typing import List

def get_requirements()->List[str]:
    """
    Returns the list of required packages.
    """
    try:
        list_requirements:List[str] = []
        with open('requirements.txt', 'r') as file:
            # Read lines from file
            lines = file.readlines()
            # Process each line
            for line_i in lines:
                # Remove spaces
                line_i = line_i.strip()
                # Ignore empty lines and "-e .""
                if line_i and line_i != '-e .':
                    list_requirements.append(line_i)
    except FileNotFoundError: 
        print("requirements.txt file not found.")

    return list_requirements

setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Farhad Dalirani",
    packages=find_packages(),
    install_requires=get_requirements()
)