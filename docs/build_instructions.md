# Building your code
Instructions on how to create distribution packages from your code

## Requirements
* pep517
* gcc (For windows mingw64 works) (only required if compiling the code)

## Setup

Set path to your gcc to CC environment variable (only required if compiling the code).

    set CC=C:\mingw64\bin\gcc.exe

## How to build your code:

The dot in this case presents the location where pyproject.toml resides

    python -m pep517.build .

For more options see:

    python -m pep517.build --help

Distribution types:
* [wheel](https://wheel.readthedocs.io/en/stable/) (binary distribution)
* [sdist](https://docs.python.org/3.6/distutils/sourcedist.html) (source distribution default gztar)
