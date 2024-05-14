import os
import h5py
import pickle
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import seaborn as sns
import scipy.io as sio
from scipy.stats import zscore
import nitime.analysis as nta
from sklearn.linear_model import Ridge, Lasso

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D

import nibabel as nib
from nilearn import image
from nilearn import datasets
from nilearn import image as nimg
from nilearn import plotting as nplot
from nilearn.plotting import plot_epi, show
from nilearn.connectome import ConnectivityMeasure

import torch
import torch.nn as nn
from typing import Optional

def save(pickle_filename:str, anything:Optional[np.ndarray]):
    """
    Pickle array

    Parameters
    ----------
    pickle_filename : str
        The filename to save the pickled array to
    anything : Optional[np.ndarray]
        The array to pickle

    Returns
    -------
    None

    """
    with open(pickle_filename, "wb") as handle:
        pickle.dump(anything, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(pickle_filename:str):
    """
    Loads a pickled array from a file.

    Parameters
    ----------
    pickle_filename : str
        The path to the pickled file to load.

    Returns
    -------
    b : Any
        The unpickled object loaded from the file.
    """
    with open(pickle_filename, "rb") as handle:
        b = pickle.load(handle)
    return b

def normalize(a:np.ndarray):
    tmp = a - np.mean(a)
    return tmp / np.std(a)

def standardize(a:np.ndarray):
    tmp = a - a.min()
    return tmp / tmp.max()

def demodulate(a:np.ndarray):
    return a / a.max()

def range11(array):
    scale = (array.max() + array.min())/2
    ret = array - scale
    return ret / array.max()

def compute_stats(vol):
    m = vol[vol!=0].mean()
    std = vol[vol!=0].std()
    return m, std

def proc_inpainted(vol, m_ref, std_ref, sparsifier=0.4):
    tmp = deepcopy(vol)
    m, std = compute_stats(tmp)
    tmp[tmp!=0] = tmp[tmp!=0] / std * std_ref
    tmp[tmp!=0] = tmp[tmp!=0] - m + m_ref
    tmp = np.abs(tmp) ** (sparsifier) * np.sign(tmp)
    return tmp

def volcoord2mnicoord(arrays, affine):
    """
    Compute volume coords to mni coords transform
    """
    tmp = np.concatenate([arrays,np.ones((arrays.shape[0],1))], axis=1)
    ret = np.matmul(affine,tmp.T)[:3].T

    return np.array(ret).astype(float)

def mnicoord2volcoord(arrays, affine):
    """
    Compute volume coords from mni coords transform
    """
    inv_affine = np.linalg.inv(affine)
    tmp = np.concatenate([arrays,np.ones((arrays.shape[0],1))], axis=1)
    ret = np.matmul(inv_affine,tmp.T)[:3].T

    return np.array(ret).astype(int)

def mean_fmri(nifti):
    """
    Compute mean (across time) fmri volume
    """

    affine = nifti.affine
    volume = nifti.get_fdata()
    mean_volume = volume.mean(axis=-1)
    ret = nib.Nifti1Image(mean_volume, affine=affine)
    return ret

def abs_nifti(nifti):
    """
    Compute absolute value nifti file
    """

    affine = nifti.affine
    volume = nifti.get_fdata()    
    ret = nib.Nifti1Image(np.abs(volume), affine=affine)
    return ret


### NI-EDU - copied code "GLM inference"
def design_variance(X, which_predictor=1):
    ''' Returns the design variance of a predictor (or contrast) in X.
    
    Parameters
    ----------
    X : numpy array
        Array of shape (N, P)
    which_predictor : int or list/array
        The index of the predictor you want the design var from.
        Note that 0 refers to the intercept!
        Alternatively, "which_predictor" can be a contrast-vector
        (which will be discussed later this lab).
        
    Returns
    -------
    des_var : float
        Design variance of the specified predictor/contrast from X.
    '''
    
    is_single = isinstance(which_predictor, int)
    if is_single:
        idx = which_predictor
    else:
        idx = np.array(which_predictor) != 0
    
    c = np.zeros(X.shape[1])
    c[idx] = 1 if is_single == 1 else which_predictor[idx]
    des_var = c.dot(np.linalg.inv(X.T.dot(X))).dot(c.T)
    return des_var