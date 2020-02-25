# Demographic Change and Human Capital in OLG Models

## Introduction

This is the code book to the term project "Demographic Change and Human Capital in OLG Models". The project aims at replicating the key features of human capital accumulation in light of demographic change presented in  Ludwig, Schelkle, Vogel (2012). The results of the term project are presented in the accompanying term paper.

The project has been created using a template for reproducible research projects in economics. Documentation on the Project Template used, on Waf, and more background can be found at https://econ-project-templates.readthedocs.io/en/stable/

## Overview

The logic of this project works by step of the analysis:

1. Data management
2. Model solution
3. Visualisation and results formatting
4. Research paper and presentations.

The code is set up such that a large variety of specifications can be run. The specifications used to create the term paper are described in more detail in the term paper itself as well as the project documentation (see bld/src/paper or bld/src/documentation, respectively).

## Running the project

Detailed information on how to prepare your system to run the project template can be found at https://econ-project-templates.readthedocs.io/en/stable/getting_started.html

In short, after cloning the repository from Github, the following steps are required to run the project (e.g. using Windows Powershell):

1. browse to the directory to which the repository has been cloned

2. create the conda virtual environment with the command:

    conda env create environment.yml

3. activate the virtual environment with the command:

    conda activate demographic_change_OLG

4. configure Waf with the command:

    python waf.py configure

5. run the build with the command:

    python waf.py build

Waf will then run all required steps to generate the outputs presented in the term paper, as well as compile the paper itself.

