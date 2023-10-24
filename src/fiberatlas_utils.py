from src.utils import *

def get_aggprop(h5dict, property):
    """
    Get the bundles statistics on whole brain level
    """
    try:
        ret = np.array(h5dict.get('matrices').get(property))
    except:
        print('Not valid property OR h5 not opened')
    return ret

def get_bundles_betweenreg(h5dict, r1, r2, verbose=True):
    """
    Get the bundles voxels spots linking two regions of interest
    """

    bundle_r = str(r1) + '_' + str(r2)

    # Check if bundle link exists
    feasible = np.array(list(set(np.array(h5dict.get('atlas')))))
    valid_regions = set([f.split('_')[0] for f in feasible] + [f.split('_')[1] for f in feasible])

    if not ((str(r1) in valid_regions) and (str(r2) in valid_regions)):
        if verbose: print('Regions inputed are not valid')
        return

    if bundle_r not in feasible:
        if verbose: print('Regions {} and {} are not sufficiently connected'.format(r1, r2))
        return
    

    ret = np.array(h5dict.get('atlas').get(bundle_r))
    return ret