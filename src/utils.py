"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

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
from scipy.stats import pearsonr

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


def volcoord2mnicoord(arrays: np.ndarray, affine: np.ndarray):
    """
    Compute volume coordinates to MNI coordinates transform.

    Transforms a set of 3D volume coordinates to MNI coordinates using 
    a provided affine transform matrix. The affine matrix maps between 
    volume voxel indices and MNI coordinates.

    Parameters
    ----------
    arrays : np.ndarray
        The volume coordinate arrays to transform. Each coordinate is a row.
    affine : np.ndarray 
        The affine transform matrix mapping volume to MNI coordinates.

    Returns
    -------
    ret : np.ndarray
        The MNI coordinates corresponding to the input volume coordinates.
    """

    tmp = np.concatenate([arrays, np.ones((arrays.shape[0], 1))], axis=1)
    ret = np.matmul(affine, tmp.T)[:3].T

    return np.array(ret).astype(float)

def mnicoord2volcoord(arrays: np.ndarray, affine: np.ndarray):
    """
    Compute volume coordinates from MNI coordinates.

    Transforms MNI coordinates to equivalent volume coordinates using
    the provided affine transform.

    Parameters
    ----------
    arrays : np.ndarray
        Array of MNI coordinates to transform.
    affine : np.ndarray
        Affine transform mapping from MNI space to volume space.

    Returns
    -------
    np.ndarray
        Array of transformed volume coordinates.
    """

    inv_affine = np.linalg.inv(affine)
    tmp = np.concatenate([arrays, np.ones((arrays.shape[0], 1))], axis=1)
    ret = np.matmul(inv_affine, tmp.T)[:3].T

    return np.array(ret).astype(int)