
""" Snob, 'because it arbitrarily puts things in classes' -- C.S. Wallace. """

import sys
from setuptools import setup, find_packages
from codecs import open
from os import path, system
from re import compile as re_compile

# For convenience.
if sys.argv[-1] == "publish":
    system("python setup.py sdist upload")
    sys.exit()

def read(filename):
    kwds = {"encoding": "utf-8"} if sys.version_info[0] >= 3 else {}
    with open(filename, **kwds) as fp:
        contents = fp.read()
    return contents

# Get the version information.
here = path.abspath(path.dirname(__file__))
vre = re_compile("__version__ = \"(.*?)\"")
version = vre.findall(read(path.join(here, "snob", "__init__.py")))[0]

setup(
    name="snob",
    version=version,
    author="Andrew R. Casey",
    author_email="andrew.casey@monash.edu",
    description="Put things in classes",
    long_description=read(path.join(here, "README.md")),
    license="MIT",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    keywords="MML snob minimum message length",
    packages=find_packages(exclude=["tests"]),
    install_requires=["numpy", "scipy", "six"],
    extras_require={
        "test": ["coverage"]
    },
    package_data={
        "": ["LICENSE"],
    },
    include_package_data=True,
    data_files=None
)
