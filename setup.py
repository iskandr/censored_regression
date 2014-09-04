import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "censored_regression",
    version = "0.0.1",
    author = "Alex Rubinsteyn",
    author_email = "alex.rubinsteyn@gmail.com",
    description = ("Linear regression where some of the labels are lower bounds"),
    license = "Apache 2.0",
    keywords = "linear regression survival censored",
    url = "http://packages.python.org/censored_regression",
    packages=['censored_regression', 'tests'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 2.7",
    ],
)