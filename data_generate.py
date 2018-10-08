import numpy as np
import networkx as nx
import scipy

def generate_random_L(p = 10,
                      a = 0.3,
                      b = 0.7,
                      diag_a = 2,
                      diag_b = 5,
                      plot = False,
                      G = nx.gn_graph(p)):
    """
    randomly generates a lower triangular matrix based on a growing network graph
    Input:
        p: number of nodes
        a: lower bound for off-diagonal
        b: upper bound for off_diagonal
        diag_a: lower bound for diagonal
        diag_b: upper bound for diagonal
        G: Directed graph
    Output:
        L: Lower triangular matrix
        A: Adjacency matrix
        G: Directed graph
    """
    ### need to relabel vertices to agree with CSCS
    mapping=dict(zip(G.nodes(),list(range(p-1,-1,-1))))
    G=nx.relabel_nodes(G,mapping)
    if(plot):
        import matplotlib.pyplot as plt
        nx.draw_shell(G, with_labels=True, font_weight='bold')
    A = nx.adjacency_matrix(G).todense()
    L = np.multiply(A,((b - a) * np.random.random_sample(size = p*p) + a).reshape(p,p))
    np.fill_diagonal(L,np.random.uniform(diag_a,diag_b,p))
    return(L,A,G)

def generate_random_MVN_data(n = 50,
                             p = 10,
                             a = 0.3,
                             b = 0.7,
                             diag_a = 2,
                             diag_b = 5,
                             plot = False):
    """generates random multivariate normal data corresponding to growing network graph
    Input:
        n: number of samples
        p: number of nodes
        a: lower bound for off-diagonal
        b: upper bound for off_diagonal
        diag_a: lower bound for diagonal
        diag_b: upper bound for diagonal
    Output:
        multivariate normal data
    """
    L,A,G = generate_random_L(p = p, a = a, b = b, diag_a = diag_a, diag_b = diag_b, plot = plot)
    omega = np.matmul(L.T,L)
    mu = np.zeros(p)
    cov = np.linalg.inv(omega)
    return(np.random.multivariate_normal(mu,cov,n))
