# !/usr/bin/env python
# -*- coding: utf-8 -*-


def main():
    from setuptools import setup, find_packages

    setup(name="y3prediction",
          version="1.0",
          description=("TBD"),
          long_description=open("README.md", "rt").read(),
          author="CEESD",
          author_email="manders2@illinois.edu",
          license="MIT",
          url="https://github.com/illinois-ceesd/drivers_y3-prediction",
          classifiers=[
              "Development Status :: 1 - Planning",
              "Intended Audience :: Developers",
              "Intended Audience :: Other Audience",
              "Intended Audience :: Science/Research",
              "License :: OSI Approved :: MIT License",
              "Natural Language :: English",
              "Programming Language :: Python",
              "Programming Language :: Python :: 3",
              "Topic :: Scientific/Engineering",
              "Topic :: Scientific/Engineering :: Information Analysis",
              "Topic :: Scientific/Engineering :: Mathematics",
              "Topic :: Scientific/Engineering :: Visualization",
              "Topic :: Software Development :: Libraries",
              "Topic :: Utilities",
              ],

          packages=find_packages(),

          python_requires="~=3.8",

          #install_requires=["mirgecom"],

          include_package_data=True,)


if __name__ == "__main__":
    main()
