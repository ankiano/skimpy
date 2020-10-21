#!/usr/bin/env python3.6
# coding=utf-8

from setuptools import setup, find_packages
from pathlib import Path
from pkg_resources import parse_requirements

with Path('requirements.txt').open() as requirements_txt:
    requirements = [
        str(requirement) for requirement in parse_requirements(requirements_txt)
    ]

setup(
    name='skimpy',
    version='0.0.10',
    description='',
    long_description='',
    py_modules=['skimpy'],
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='skimpy',
)
