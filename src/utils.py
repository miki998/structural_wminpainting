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
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D

import nibabel as nib
from nilearn import datasets
from nilearn import image as nimg
from nilearn import plotting as nplot


### saving and loading made-easy
def save(pickle_file, array):
    """
    Pickle array
    """
    with open(pickle_file, 'wb') as handle:
        pickle.dump(array, handle, protocol=pickle.HIGHEST_PROTOCOL)
def load(pickle_file):
    """
    Loading pickled array
    """
    with open(pickle_file, 'rb') as handle:
        b = pickle.load(handle)
    return b

def standardize(a):
    tmp = a - a.min()
    return tmp / tmp.max()

def demodulate(a):
    return a/a.max()