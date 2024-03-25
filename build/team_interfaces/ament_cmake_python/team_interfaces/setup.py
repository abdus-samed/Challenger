from setuptools import find_packages
from setuptools import setup

setup(
    name='team_interfaces',
    version='0.0.0',
    packages=find_packages(
        include=('team_interfaces', 'team_interfaces.*')),
)
