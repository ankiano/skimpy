#!/usr/bin/env python3.6
# coding=utf-8

try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements
try: # for pip >= 10
    from pip._internal.download import PipSession
except ImportError: # for pip <= 9.0.3
    from pip.download import PipSession

from setuptools import setup, find_packages

parsed_requirements = parse_requirements(
    'requirements.txt',
    session=PipSession()
)

requirements = [str(ir.req) for ir in parsed_requirements]

setup(
    name='skimpy',
    version='0.0.9',
    description='',
    long_description='',
    py_modules=['skimpy'],
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='skimpy',
)
