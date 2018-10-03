#!/usr/bin/env python3.6
# coding=utf-8

import skimpy

from pip._internal.req import parse_requirements
from pip._internal.download import PipSession

from setuptools import setup, find_packages

parsed_requirements = parse_requirements(
    'requirements.txt',
    session=PipSession()
)

requirements = [str(ir.req) for ir in parsed_requirements]

setup(
    name='skimpy',
    version=skimpy.__version__,
    description='',
    long_description='',
    py_modules=['skimpy'],
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='skimpy',
)
