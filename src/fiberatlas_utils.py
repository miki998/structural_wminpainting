from src.utils import *

def get_aggprop(h5dict: h5py._hl.files.File, property: str):
    """
    Get the bundles statistics on whole brain level from the HDF5 file.

    Parameters
    ----------
    h5dict : h5py._hl.files.File 
        The opened HDF5 file.
    property : str
        The property to extract from the HDF5 file.

    Returns
    -------
    ret : np.arrasy
        The array containing the requested property values.
    """

    try:
        ret = np.array(h5dict.get("matrices").get(property))
    except:
        print("Not valid property OR h5 not opened")
    return ret


def get_bundles_betweenreg(h5dict: h5py._hl.files.File, r1: int, r2: int, verbose=True):
    """
    Get the voxels for the bundle linking two regions of interest.

    Parameters
    ----------
    h5dict : h5py._hl.files.File
        The HDF5 file containing the fiber atlas data.
    r1 : int 
        The index of the first region of interest.
    r2 : int
        The index of the second region of interest.  
    verbose : bool, optional
        Whether to print status messages, by default True

    Returns
    -------
    ret : np.ndarray
        The voxel indices for the bundle linking r1 and r2, 
        or None if the regions are not connected.

    """

    bundle_r = str(r1) + "_" + str(r2)

    # Check if bundle link exists
    feasible = np.array(list(set(np.array(h5dict.get("atlas")))))
    valid_regions = set(
        [f.split("_")[0] for f in feasible] + [f.split("_")[1] for f in feasible]
    )

    if not ((str(r1) in valid_regions) and (str(r2) in valid_regions)):
        if verbose:
            print("Regions inputed are not valid")
        return

    if bundle_r not in feasible:
        if verbose:
            print("Regions {} and {} are not sufficiently connected".format(r1, r2))
        return

    ret = np.array(h5dict.get("atlas").get(bundle_r))
    return ret
