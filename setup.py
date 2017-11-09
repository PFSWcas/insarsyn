#!/usr/bin/python3
from setuptools import setup

setup(
    name="insarsyn",
    version="0.2",
    description="synthetic test data for SAR interferometry",
    author="Gerald Baier",
    author_email="gerald.baier@tum.de",
    zip_safe=False,
    py_modules=['insarsyn', 'patterns', 'fractals', 'backscatter', 'coherence', 'stack'],
    install_requires=['numpy>=1.8.1', 'scipy>=0.13.3']
)
