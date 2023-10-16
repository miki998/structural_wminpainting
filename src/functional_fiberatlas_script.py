# coding: utf-8
import numpy as np

import scipy.io as sio
from scipy.stats import zscore
import matplotlib.pyplot as plt

import h5py
from copy import deepcopy
from tqdm import tqdm


import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import nibabel as nib
from nilearn import datasets
from nilearn import image as nimg
from nilearn import plotting as nplot
import numpy as np

import scipy.io as sio
from scipy.stats import zscore
import matplotlib.pyplot as plt

import h5py
from copy import deepcopy
from tqdm import tqdm


import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import nibabel as nib
from nilearn import datasets
from nilearn import image as nimg
from nilearn import plotting as nplot
def get_aggprop(h5dict, property):
    """
    Get the bundles statistics on whole brian level
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
connFilename = '../atlas_data/fiber_atlas/probconnatlas/wm.connatlas.scale1.h5'
hf = h5py.File(connFilename, 'r')
connFilename = '../atlas_data/fiber_atlas/probconnatlas/wm.connatlas.scale1.h5'
hf = h5py.File(connFilename, 'r')
centers = np.array(hf.get('header').get('gmcoords'))
# Load the rest fmri in MNI space volumes
rst_vol = nib.load('../atlas_data/rstfMRI_eg/filtered_func_data_res_MNI.nii.gz')
consistency_view = get_aggprop(hf, 'consistency')
length_view = get_aggprop(hf, 'length')
nbStlines_view = get_aggprop(hf, 'numbStlines')
# Ploting the stats view
fig, ax = plt.subplots(1,3, figsize=(15,5))
im0 = ax[0].imshow(consistency_view)
im1 = ax[1].imshow(length_view)
im2 = ax[2].imshow(nbStlines_view)
ax[0].set_title('Consistency of appearance across subjects')
ax[1].set_title('Length of streamlines between regions')
ax[2].set_title('Number of Streamlines between regions')
fig.colorbar(im0,  orientation='horizontal')
fig.colorbar(im1,  orientation='horizontal')
fig.colorbar(im2,  orientation='horizontal')
streamline = get_bundles_betweenreg(hf, 3,78)
path_motion = []
coords = streamline[:,[0,1,2]]
for k in range(streamline.shape[0]-1):
    dist = np.sqrt(np.sum((coords[k] - coords[k-1]) ** 2))
    path_motion.append(dist)

fig ,ax = plt.subplots(1,3, figsize=(20,5))
ax[0].hist(streamline[:,3], label='# Subject with intersected bundles')
ax[0].set_title('# of subjects with the specific bundle')
ax[0].legend()
ax[1].plot(path_motion)
ax[2].scatter(coords[:,0],coords[:,1])
# plt.xlim(0,100)
fig,axs = plt.subplots(1,1,subplot_kw=dict(projection='3d'),figsize=(5,5))

length_order = 200
x=coords[:,0][:length_order]
y=coords[:,1][:length_order]
z=coords[:,2][:length_order]

axs.plot(x,y,z)
axs.set_title('3d viz')
plt.show()
from nilearn import image
from tqdm import tqdm

# 1. Transform the voxels to coordinates
# 2. Match the coordinates i.e the voxels to a given label
# 3. Obtain an mapping of a voxel to the parcels

labels_vol = np.zeros(rst_vol.shape[:3])
for x in tqdm(range(rst_vol.shape[0])):
    for y in range(rst_vol.shape[1]):
        for z in range(rst_vol.shape[2]):
            # Find the MNI coordinates of the voxel (x, y, z)
            spatial_coord = np.array(image.coord_transform(x, y, z, rst_vol.affine))
            labels_vol[x,y,z] = np.argmin(np.sum((centers - spatial_coord) ** 2, axis=1))

# Atlas averaging timecourses
avg_tc = np.zeros((centers.shape[0], rst_vol.shape[-1]))
for t in tqdm(range(rst_vol.shape[-1])):
    tmp = rst_vol.get_fdata()[:,:,:,t]
    for k in range(centers.shape[0]):
        indexes = (labels_vol == k)
        nonzero = tmp[indexes][(tmp[indexes] != 0)]
        avg_tc[k,t] = nonzero.mean()
import nitime.analysis as nta

corr_mat = np.zeros((avg_tc.shape[0],avg_tc.shape[0]))
for k in range(avg_tc.shape[0]):
    corr_mat[k] = nta.SeedCorrelationAnalyzer(avg_tc[k], avg_tc).corrcoef
import numpy as np

import scipy.io as sio
from scipy.stats import zscore
import matplotlib.pyplot as plt

import h5py
from copy import deepcopy
from tqdm import tqdm


import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import nibabel as nib
from nilearn import datasets
from nilearn import image as nimg
from nilearn import plotting as nplot

import nitime.analysis as nta
import numpy as np

import scipy.io as sio
from scipy.stats import zscore
import matplotlib.pyplot as plt

import h5py
from copy import deepcopy
from tqdm import tqdm


import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import nibabel as nib
from nilearn import datasets
from nilearn import image as nimg
from nilearn import plotting as nplot

import nitime.analysis as nta
corr_mat = np.zeros((avg_tc.shape[0],avg_tc.shape[0]))
for k in range(avg_tc.shape[0]):
    corr_mat[k] = nta.SeedCorrelationAnalyzer(avg_tc[k], avg_tc).corrcoef
plt.imshow(corr_mat)
plt.colorbar()
# Compute which bundles have intersecting voxels, basically a intersection matrix would be our X samples matrix
allstreamlines = []
actualstreamlines = []
for i in tqdm(range(1,96)):
    for j in range(i+1,96):
        tmp = get_bundles_betweenreg(hf, i, j, verbose=False)
        if tmp is None:
            allstreamlines.append(tmp)
        else:
            allstreamlines.append(tmp[:,[0,1,2]])
            actualstreamlines.append(tmp[:,[0,1,2]])
voxels = set(tuple(map(tuple, actualstreamlines[0])))
for k in tqdm(range(len(actualstreamlines))):
    voxels = voxels.union(set(tuple(map(tuple, actualstreamlines[k]))))
S = []
for k in tqdm(range(len(allstreamlines))):
    if not (allstreamlines[k] is None):
        S.append(set(tuple(map(tuple, allstreamlines[k]))))
    else:
        S.append(None)
streamline_conn = np.zeros((len(allstreamlines),len(allstreamlines)))
for k in tqdm(range(len(allstreamlines))):
    # for each streamline look for all the other streamlines that might have intersecting voxels and if yes then flip
    template = S[k]
    if template is None:
        streamline_conn[k,:] = np.nan
        continue
    for m in range(len(allstreamlines)):
        database = S[m]
        if database is None:
            streamline_conn[k,m] = np.nan
            continue
        streamline_conn[k,m] = len(template.intersection(database))
print()
streamline_conn
plt.imshow(streamline_conn)
from sklearn.linear_model import Ridge

clf = Ridge(alpha=1.0)
clf.fit(X, y)
y.shape
# Generating the X samples

# Generating the y samples
# 1. Careful as well to remove the auto-correlation in the diagonal
# 2. Raster scan parsing meaning that it is the activity of (R0,R1) -> (R0,R2) -> (R0,R3) etc...
y = corr_mat[np.triu_indices(corr_mat.shape[0], 1)]
streamline_conn
streamline_conn.shape
streamline_conn[0]
np.isnan(streamline_conn[0])
np.sum(np.isnan(streamline_conn[0]))
np.isnan(streamline_conn[0])
plt.hist(streamline_conn[0])
streamline_conn[0] == 0
np.sum(streamline_conn[0] == 0)
plt.imshow(np.nan_to_num(streamline_conn))
# 1. Normalize count
X = np.nan_to_num(streamline_conn)

# 2. Binarize it by thresholding
# 1. Normalize count
X = deepcopy(np.nan_to_num(streamline_conn))

# 2. Binarize it by thresholding
from stats import zscore
from scipy.stats import zscore
# 1. Normalize count
X = deepcopy(np.nan_to_num(streamline_conn))
X = zscore(X)

# 2. Binarize it by thresholding
plt.imshow(X)
zscore(X
zscore(X)
# 1. Normalize count
X = zscore(np.nan_to_num(streamline_conn))
# X = zscore(X)

# 2. Binarize it by thresholding
from scipy.stats import zscore
plt.imshow(X)
X.min(axis=0)
X.min(axis=0).shape
# 1. Standardize count
X = np.nan_to_num(streamline_conn)
# X = X - X.min(axis=0)

# 2. Binarize it by thresholding
X.min(axis=0).shape
X.min(axis=0)
X
X.max(axis=0)
X/X.max(axis=0)
X.max(axis=0) == 0
np.sum(X.max(axis=0) == 0)
# 1. Standardize count
X = np.nan_to_num(streamline_conn)
X = X/X.max(axis=0)
X = np.nan_to_num(X)

# 2. Binarize it by thresholding
plt.imshow(X)
plt.imshow(X > 0.5)
# Example of connectivity we will be using as a set of y samples
plt.imshow(np.triu(corr_mat, 1))
# Example of connectivity we will be using as a set of y samples
plt.imshow(np.triu(corr_mat, 1))
# 1. Standardize count
X = np.nan_to_num(streamline_conn)
X = X/X.max(axis=0)
X = np.nan_to_num(X)

plt.imshow(X)
plt.show()
# 2. Binarize it by thresholding
X = 
# 1. Standardize count
X = np.nan_to_num(streamline_conn)
X = X/X.max(axis=0)
X = np.nan_to_num(X)

plt.imshow(X)
plt.show()

# 2. Binarize it by thresholding
X = (X > 0.5).astype(float)
# Example of connectivity we will be using as a set of y samples
fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].imshow(np.triu(corr_mat, 1))
ax[1].imshow(X)
from sklearn.linear_model import Ridge

clf = Ridge(alpha=1.0)
clf.fit(X, y)
clf.coef_
clf.coef_.shape
clf.coef_
plt.plot(clf.coef_)
clf.predict(X)
y
plt.plot(y)
plt.plot(clf.predict(X))
plt.plot(y)
plt.plot(clf.predict(X))
from scipy.stats import pearsonr
pearsonr(y, clf.predict(X))
plt.plot(clf.coef_)
coef_.shape
clf.coef_
clf.coef_.shape
clf.coef_
from scipy.stats import pearsonr
print(
import numpy as np

import scipy.io as sio
from scipy.stats import zscore
import matplotlib.pyplot as plt

import h5py
from copy import deepcopy
from tqdm import tqdm


import seaborn as sns
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import nibabel as nib
from nilearn import datasets
from nilearn import image as nimg
from nilearn import plotting as nplot

import nitime.analysis as nta
streamline_conn
get_ipython().run_line_magic('save', '(streamline_conn)')
get_ipython().run_line_magic('save', '')
