from setuptools import find_packages,setup
from typing import List


HYPHEN_E_DOT='-e .'
def get_requirement(file_path:str)->List:
    """
    This function will return the list of requirements
    """
    requirement=[]
    with open(file_path,'r') as fobj:
        requirement=fobj.readlines()
        requirement=[req.replace("\n","") for req in requirement]

        if HYPHEN_E_DOT in requirement:
            requirement.remove(HYPHEN_E_DOT)
    return requirement
setup(
name="ML Project",
version='0.0.1',
author="Kartik Mehta",
author_email="kpmehta7806@gmail.com",
packages=find_packages(),
# install_requires=["pandas",'numpy','seaborn']
install_requires=get_requirement("requirements.txt")
)

