import numpy as np
import networkx as nx
from data_generate import generate_random_MVN_data
from CSCS import CSCS

if __name__ == '__main__':
    np.random.seed(3689)
    print('Generating random data ... ')
    Y = generate_random_MVN_data()
    print('Done!')
    print('Fitting CSCS ...')
    cscs = CSCS(Y = Y,l = 1)
    L,A,G = cscs.fit()
    print('Done!')