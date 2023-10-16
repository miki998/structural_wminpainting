from src.utils import *
from scipy.stats import pearsonr

def TV(gft_signal):
    """
    Compute total variation in spectral domain 
    Note: if orthogonal eigenvectors this is equivalent to TV in node domain
    """
    return np.sqrt(np.abs(gft_signal @ gft_signal))

def TV_sig(signal):
    """
    Compute total variation in node domain -> Sum of energy
    """
    return np.sqrt(np.abs(signal @ signal))
