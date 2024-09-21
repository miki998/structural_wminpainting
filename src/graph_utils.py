"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from src.utils import *


def make_cycle(N:int):
    """
    Return adjacency matrix of a chain graph of length N
    Parameters
    ----------
    N : int
        Number of nodes

    Returns
    -------
    A : np.ndarray
        The adjacency matrix reprensenting the cycle
    """
        
    A = np.diag(np.ones(N-1))
    A = np.concatenate([A,np.zeros((1,N-1))])
    bound = np.zeros((N,1))
    bound[-1] = 1.0
    A = np.concatenate([bound, A], axis=1)
    return A


def normalize_adjacency(A: np.ndarray, tnorm="right"):
    """
    Normalize the adjacency matrix by in-degrees / out-degrees / symmetric

    Parameters
    ----------
    A : np.ndarray
        The adjacency matrix to normalize
    tnorm : str
        The normalization method. Can be "right", "left", or "symmetric".

    Returns
    -------
    normA : np.ndarray
        The normalized adjacency matrix
    """

    if tnorm == "right":
        indegrees = np.sum(A, axis=0)
        factors_in = np.diag(np.divide(1, indegrees, where=np.abs(indegrees) > 1e-10))
        normA = A @ factors_in

    if tnorm == "left":
        outdegrees = np.sum(A, axis=1)
        factors_out = np.diag(
            np.divide(1, outdegrees, where=np.abs(outdegrees) > 1e-10)
        )
        normA = factors_out @ A

    if tnorm == "symmetric":
        indegrees = np.sum(A, axis=0)
        outdegrees = np.sum(A, axis=1)

        if (np.sum(indegrees < 0) + np.sum(outdegrees < 0)) > 0:
            print("Negative Degrees")
            return

        indegrees = np.sqrt(indegrees)
        outdegrees = np.sqrt(outdegrees)

        factors_in = np.diag(np.divide(1, indegrees, where=np.abs(indegrees) > 1e-10))
        factors_out = np.diag(
            np.divide(1, outdegrees, where=np.abs(outdegrees) > 1e-10)
        )
        normA = factors_out @ A @ factors_in

    return normA
    return normA
    

def compute_directed_laplacian(A:np.ndarray):
    """
    Compute the directed Laplacian matrix for a given adjacency matrix A. 

    The directed Laplacian is defined as L = D - A, where D is a diagonal matrix containing the in-degree of each node, and A is the adjacency matrix.

    Parameters
    ----------
    A : ndarray
        Adjacency matrix  

    Returns
    -------
    L : ndarray
        Directed Laplacian matrix
    """

    # Compute indegrees
    indeg = A.sum(axis=0).astype(float)
    ret = np.diag(indeg) - A.astype(float)

    return ret
