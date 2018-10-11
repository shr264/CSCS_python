import numpy as np
import numba
from numba import jit
import networkx as nx
from sys import exit
from joblib import Parallel, delayed
import multiprocessing

def _soft_threshold(x,
                    l):
    """
    soft thresholding function
    Inputs:
        x: real number
        l: soft thresholding values
    output:
        soft thresholded value
    """
    return(np.sign(x)*max(abs(x)-l,0))

def _Tj(j,
        A,
        x,
        l):
    """
    look at the algorithm in the paper for more details
    Inputs:
        j: index at which to apply function
        A: a p x p matrix
        x: a p x 1 vector
        l: real number
    output:
        an updated real value
    """
    return(_soft_threshold(sum(np.delete(A[:,j], j, axis=0)*np.delete(x, j, axis=0)),l)/(2*A[j,j]))

def _Tk(k,
        A,
        x):
    """
    look at the algorithm in the paper for more details
    Inputs:
        k: index at which to apply function
        A: a p x p matrix
        x: a p x 1 vector
    output:
        an updated real value
    """
    sum_term = sum(np.delete(A[:,k], k, axis=0)*np.delete(x, k, axis=0))
    return((-sum_term+np.sqrt(np.square(sum_term)+4*A[k,k]))/(2*A[k,k]))

@jit(nopython=True)
def _soft_threshold_numba(x,l):
    """
    soft thresholding function
    Inputs:
        x: real number
        l: soft thresholding values
    output:
        soft thresholded value
    """
    return(np.sign(x)*max(abs(x)-l,0))

@jit(nopython=False)
def _Tj_numba(j,
        A,
        x,
        l):
    """
    look at the algorithm in the paper for more details
    Inputs:
        j: index at which to apply function
        A: a p x p matrix
        x: a p x 1 vector
        l: real number
    output:
        an updated real value
    """
    return(_soft_threshold_numba(sum(np.delete(A[:,j], j, axis=0)*np.delete(x, j, axis=0)),l)/(2*A[j,j]))

@jit(nopython=False)
def _Tk_numba(k,
        A,
        x):
    """
    look at the algorithm in the paper for more details
    Inputs:
        k: index at which to apply function
        A: a p x p matrix
        x: a p x 1 vector
    output:
        an updated real value
    """
    sum_term = sum(np.delete(A[:,k], k, axis=0)*np.delete(x, k, axis=0))
    return((-sum_term+np.sqrt(np.square(sum_term)+4*A[k,k]))/(2*A[k,k]))

def _hk(k,
        A,
        l,
        maxitr=100,
        tol=1e-5,
        debug = False):
    """
    look at the algorithm in the paper for more details
    Inputs:
        k: index at which to apply function
        A: a p x p matrix
        l: real value to threshold by
        maxitr: maximum number of iterations to run
        tol: error tolerance
        debug: print outputs used in debugging
    output:
        an updated real vector
    """
    p = A.shape[0]
    xold = np.zeros(p)
    xold[k] = 1
    xnew = np.copy(xold)
    r = 1
    converged = False
    while(converged is False):
        if(debug):
            print('iter: {}'.format(r))
        for j in range(k):
            xnew[j] = _Tj(j,A,xnew,l)
        xnew[k] = _Tk(k,A,xnew)
        if(max(abs(xnew - xold))<tol):
            if(debug):
                print('Tolerance condition met')
            converged = True
        elif(r>maxitr):
            if(debug):
                print('Algorithm did not converge within 100 reps')
            converged = True
        else:
            if(debug):
                print('Conditions not met. Increasing iteration count')
            r += 1
        xold = xnew
    return(xnew)

@jit(nopython=False)
def _hk_numba(k,
        A,
        l,
        maxitr=100,
        tol=1e-5,
        debug = False):
    """
    look at the algorithm in the paper for more details
    Inputs:
        k: index at which to apply function
        A: a p x p matrix
        l: real value to threshold by
        maxitr: maximum number of iterations to run
        tol: error tolerance
        debug: print outputs used in debugging
    output:
        an updated real vector
    """
    p = A.shape[0]
    xold = np.zeros(p)
    xold[k] = 1
    xnew = np.copy(xold)
    r = 1
    converged = False
    while(converged is False):
        #if(debug):
        #    print('iter: {}'.format(r))
        for j in range(k):
            xnew[j] = _Tj_numba(j,A,xnew,l)
        xnew[k] = _Tk_numba(k,A,xnew)
        if(max(abs(xnew - xold))<tol):
        #    if(debug):
        #        print('Tolerance condition met')
            converged = True
        elif(r>maxitr):
        #    if(debug):
        #        print('Algorithm did not converge within 100 reps')
            converged = True
        else:
        #    if(debug):
        #        print('Conditions not met. Increasing iteration count')
            r += 1
        xold = xnew
    return(xnew)

def _hk_par(k,
        A,
        p,
        l,
        maxitr=100,
        tol=1e-5,
        debug = False):
    """
    look at the algorithm in the paper for more details
    Inputs:
        k: index at which to apply function
        A: a p x p matrix
        l: real value to threshold by
        maxitr: maximum number of iterations to run
        tol: error tolerance
        debug: print outputs used in debugging
        num_workers: number of cores to use
    output:
        an updated real vector
    """
    xold = np.zeros(p)
    xold[k] = 1
    xnew = np.copy(xold)
    r = 1
    converged = False
    while(converged is False):
        if(debug):
            print('iter: {}'.format(r))
        for j in range(k):
            xnew[j] = _Tj(j,A,xnew[0:(k+1)],l)
        xnew[k] = _Tk(k,A,xnew[0:(k+1)])
        if(max(abs(xnew - xold))<tol):
            if(debug):
                print('Tolerance condition met')
            converged = True
        elif(r>maxitr):
            if(debug):
                print('Algorithm did not converge within 100 reps')
            converged = True
        else:
            if(debug):
                print('Conditions not met. Increasing iteration count')
            r += 1
        xold = xnew
    return(xnew)

@jit(nopython=False)
def CSCS_numba_fit(Y,
         l,
         maxitr=100,
         tol=1e-4,
         warmstart=False,
         debug = False):
    """
    implements the CSCS algorithm from
    A convex framework for high-dimensional sparse Cholesky based covariance estimation
    by
    Kshitij Khare, Sang Oh, Syed Rahman and Bala Rajaratnam
    https://arxiv.org/pdf/1610.02436.pdf
    #### inputs
    ## Y: n by p matrix of data
    ## lambda: l1-penalty parameter
    ## maxitr: maximum number of iterations allowed (diagonal/off-diagonal optimization)
    ## tol: if maximum difference of iterates is less than tol, consider converged
    ## warmstart: warmstarting actually made the runs slower (overhead may be too expensive)

    #### outputs
    ## L: lower triangular matrix of cholesky factor computed with CSCS algorithm
    ## A: adjacency matrix for graph
    ## D: directed graph
    """
    n, p = Y.shape
    L = np.identity(p)
    S = (Y.T@Y)/n
    L[0,0] = 1/np.sqrt(S[0,0])
    for i in range(1,p):
        #if(debug):
        #    print('Variable: {}'.format(i+1))
        L[i,0:(i+1)] = _hk_numba(i,S[0:(i+1), 0:(i+1)], l, maxitr, tol, debug = debug)
    A = (L!=0)*1.0
    G=nx.from_numpy_matrix(A.T, create_using=nx.DiGraph())
    return(L, A, G)

def CSCS_fit(Y,
         l,
         maxitr=100,
         tol=1e-4,
         warmstart=False,
         debug = False):
    """
    implements the CSCS algorithm from
    A convex framework for high-dimensional sparse Cholesky based covariance estimation
    by
    Kshitij Khare, Sang Oh, Syed Rahman and Bala Rajaratnam
    https://arxiv.org/pdf/1610.02436.pdf
    #### inputs
    ## Y: n by p matrix of data
    ## lambda: l1-penalty parameter
    ## maxitr: maximum number of iterations allowed (diagonal/off-diagonal optimization)
    ## tol: if maximum difference of iterates is less than tol, consider converged
    ## warmstart: warmstarting actually made the runs slower (overhead may be too expensive)

    #### outputs
    ## L: lower triangular matrix of cholesky factor computed with CSCS algorithm
    ## A: adjacency matrix for graph
    ## D: directed graph
    """
    n, p = Y.shape
    L = np.identity(p)
    S = np.matmul(Y.T,Y)/n
    L[0,0] = 1/np.sqrt(S[0,0])
    for i in range(1,p):
        if(debug):
            print('Variable: {}'.format(i+1))
        L[i,0:(i+1)] = _hk(i,S[0:(i+1), 0:(i+1)], l, maxitr, tol, debug = debug)
    A = (L!=0)*1.0
    G=nx.from_numpy_matrix(A.T, create_using=nx.DiGraph())
    return(L, A, G)

def CSCS_par_fit(Y,
         l,
         maxitr=100,
         tol=1e-4,
         warmstart=False,
         debug = False,
         num_workers = None):
    """
    implements the CSCS algorithm in parallel from
    A convex framework for high-dimensional sparse Cholesky based covariance estimation
    by
    Kshitij Khare, Sang Oh, Syed Rahman and Bala Rajaratnam
    https://arxiv.org/pdf/1610.02436.pdf
    #### inputs
    ## Y: n by p matrix of data
    ## lambda: l1-penalty parameter
    ## maxitr: maximum number of iterations allowed (diagonal/off-diagonal optimization)
    ## tol: if maximum difference of iterates is less than tol, consider converged
    ## warmstart: warmstarting actually made the runs slower (overhead may be too expensive)

    #### outputs
    ## L: lower triangular matrix of cholesky factor computed with CSCS algorithm
    ## A: adjacency matrix for graph
    ## D: directed graph
    """
    n, p = Y.shape
    S = np.matmul(Y.T,Y)/n
    if(num_workers is None):
        num_workers = multiprocessing.cpu_count() - 1
    L = np.vstack(Parallel(n_jobs=num_workers)(delayed(_hk_par)(i,
                                                      S[0:(i+1),
                                                      0:(i+1)],
                                                      p = p,
                                                      l = l,
                                                      maxitr = maxitr,
                                                      tol = tol,
                                                      debug = debug)
                                                      for i in range(p)))


    A = (L!=0)*1.0
    G=nx.from_numpy_matrix(A.T, create_using=nx.DiGraph())
    return(L, A, G)

class CSCS:
    def __init__(self,
                 Y = None,
                 l = None,
                 num_workers = 1):
        self.Y = Y
        self.l = l
        self.num_workers  = num_workers

    def fit(self,
            maxitr=100,
            tol=1e-4,
            warmstart=False,
            debug = False):
        if(self.Y is None):
            print('Please enter a data matrix')
            exit(1)
        if(self.l is None):
            print('Please enter a thresholding value')
            exit(1)
        if(self.num_workers>1):
            return(CSCS_par_fit(self.Y,
                       self.l,
                       maxitr=maxitr,
                       tol=tol,
                       warmstart=warmstart,
                       debug = debug,
                       num_workers = self.num_workers))
        elif(self.num_workers==1):
            return(CSCS_fit(self.Y,
                       self.l,
                       maxitr=maxitr,
                       tol=tol,
                       warmstart=warmstart,
                       debug = debug))
        else:
            print('Incorrect num_workers! Please enter a positive integer')
            exit(1)
