# CSCS

This implements the CSCS algorithm for covariance and DAG estimation from [A convex framework for high-dimensional sparse Cholesky based covariance estimation](https://arxiv.org/pdf/1610.02436.pdf) by Kshitij Khare, Sang Oh, Syed Rahman and Bala Rajaratnam

## Basic scripts

The program consists of the following scripts
* data_generate.py: used to generate random multivariate data accoriding to a graph
* CSCS.py: contains the main functions and class for CSCS
* main.py: runs the program to generate the DAG

## Notebooks

A notebook with an example is also included
* CSCS_in_python_example.ipynb

## Example

```
import numpy as np
import networkx as nx
from data_generate import generate_random_MVN_data
from CSCS import CSCS

np.random.seed(3689)
Y = generate_random_MVN_data()
cscs = CSCS(Y = Y,l = 1)
L,A,G = cscs.fit()
```

## Authors

* **Syed Rahman**
