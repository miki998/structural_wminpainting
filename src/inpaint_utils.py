from ctypes import Union
from src.utils import *
from src.fiberatlas_utils import *

def property_regularizer(x, C, norm="L2"):
    """
    Compute a regularizer value for a bundle property array x 
    based on the given weighting array C and norm.

    Lower values in C will penalize changes in x more strongly.

    Parameters
    ----------
    x : ndarray
        The bundle property array.
    C : ndarray
        The weighting array.
    norm : str
        The norm type to use. Can be 'L2', 'L1' or 'PL2'.

    Returns
    ------- 
    ret : float
        The computed regularizer value.

    """
    inv_C = 1 / C
    if norm == "L2":
        ret = ((inv_C * x) ** 2).sum() / C.shape[0]
    elif norm == "L1":
        ret = ((inv_C * x).abs()).sum() / C.shape[0]
    elif norm == "PL2":
        # inv_C += 0.1 # buffer infinite
        ret = (x.abs() ** inv_C).sum() / C.shape[0]
    else:
        print("Unsupported Norm")
    return ret


def regress_linear(X, y, lbd, ridgeflag=True, verbose=False):
    # NOTE: y is of dimension (#regions,#TR)
    """
    Applying linear regression with given design matrix
    with either Lasso or Ridge regularization

    UNIT-TEST (TODO)

    scalenorm = np.abs(region_ftimecourse).mean()
    plt.hist(np.array(perf_lasso['scores'])[:,0] / scalenorm, alpha=0.3)
    plt.hist(np.array(perf_lasso['scores'])[:,1] / scalenorm, alpha=0.3)
    plt.hist(np.array(perf_lasso['scores'])[:,2] / scalenorm, alpha=0.3)
    plt.hist(np.array(perf_lasso['scores'])[:,3] / scalenorm, alpha=0.3)
    plt.hist(np.array(perf_lasso['scores'])[:,5] / scalenorm, alpha=0.3)

    plt.xscale('log')

    Parameters
    ----------
    X : ndarray
        Design matrix with dimensions (#features, #samples) 
    y : ndarray 
        Response variable with dimensions (#regions, #timepoints)
    lbd : float
        Regularization strength 
    ridgeflag : bool
        Whether to use Ridge (True) or Lasso (False) regularization
    verbose : bool
        Whether to print progress 

    Returns
    -------
    coefs : ndarray
        Learned coefficients for each timepoint, (#features, #timepoints)
    scores : ndarray 
        Mean squared error on training set for each timepoint
    intercepts : ndarray
        Learned intercepts for each timepoint

    """

    coefs = []
    scores = []
    intercepts = []
    for tidx in tqdm(range(y.shape[1]), disable=not verbose):
        yt = y[:, tidx]

        if ridgeflag:
            clf = Ridge(alpha=lbd)
        else:
            clf = Lasso(alpha=lbd)

        clf.fit(X.T, yt)
        scores.append(((clf.predict(X.T) - yt) ** 2).mean())
        coefs.append(clf.coef_)
        intercepts.append(clf.intercept_)

    coefs = np.array(coefs)
    scores = np.array(scores)
    intercepts = np.array(intercepts)

    return coefs, scores, intercepts


######### NOTE: REMOVE IF NOT USED IN THE END FOR SATURATION ##########
def scaled_sigmoid(x: np.ndarray, smin: float, smax: float, 
                   saturation_point:float=0.95, eps:float=0.2):
    """
    Apply scaled sigmoid on an array with numpy

    UNIT-TEST (TODO)

    sat_min, sat_max = -100, 40
    tmp = np.linspace(sat_min-50, sat_max+50, 1000)
    plt.plot(tmp, scaled_sigmoid(tmp, sat_min, sat_max, saturation_point=0.95))
    plt.hlines(-120, xmin=-150, xmax=100, linestyle='--', color='k')
    plt.hlines(48, xmin=-150, xmax=100, linestyle='--', color='k')
    plt.vlines(-150, ymin=-125, ymax=50, linestyle='--', color='r')
    plt.vlines(100, ymin=-125, ymax=50, linestyle='--', color='r')

    # Small check for the distribution of min max activity
    tmp = np.array([[region_ftimecourse[k].min(), region_ftimecourse[k].max()] for k in range(len(region_ftimecourse))])
    plt.scatter(np.abs(tmp[:,0]), tmp[:,1])

    Parameters
    ----------

    Returns
    -------
    fx : 

    """

    b = smax - (smax - smin)/2
    torchflag = type(x) is torch.Tensor

    a = (b - smax) / np.log(1/saturation_point - 1)

    sx = (b-x)/a
    A = smin + smin * eps
    K = smax + smax * eps
    if torchflag:
        fx = A + (K-A) / (1 + torch.exp(sx))
    else:
        fx = A + (K-A) / (1 + np.exp(sx))

    return fx

def optimize_lreg(X, y, Ls, Lt, num_epochs=1000, 
                  lr=1e-3, seed=99, verbose=False, p1=10.0,
                    p2=10.0, logging=False, saturate=False, 
                    saturate_factor=0.95, positive_constraint=False):
    """
    Optimize linear regression parameters with property based regularizers.

    Parameters
    ----------
    - y: Full fMRI timecourse for all regions (#regions, #timepoints) 
    - X: Bundle encoding matrix (#bundles, #regions)
    - Ls: Spatial Laplacian across bundles
    - Lt: Temporal Laplacian across regions at bundles
    - num_epochs: Number of training epochs
    - lr: Learning rate 
    - seed: Random seed 
    - verbose: Print progress
    - p1: Weight for spatial regularizer
    - p2: Weight for temporal regularizer


    Returns
    -------
    - pred_activity: Predicted bundle activities
    - logging_dict: Dictionary containing loss values per epoch (optional)
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    ywhite_init = np.random.random((y.shape[1], X.shape[-1])) - 0.5

    # Converting np arrays into Torchs
    X = torch.Tensor(X)
    y = torch.Tensor(y.T)
    ywhite = torch.Tensor(ywhite_init).requires_grad_(True)
    Ls = torch.Tensor(Ls)
    Lt = torch.Tensor(Lt)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam([ywhite], lr=lr)

    if logging:
        logging_dict = {'gen_loss': [],'spatial_reg_loss': [], 'temporal_reg_loss': [], 'positiveness_reg_loss': [], 'total_loss': []}
    if saturate:
        lbound, rbound = y.min(axis=1).values, y.max(axis=1).values

    for _ in tqdm(range(num_epochs), disable=(not verbose)):
        # Looping within epochs across time and space
        
        # Compute datafit loss (saturate flag -> allows for saturation of individual bundles' activities)
        generic_loss = torch.zeros(1)
        for tidx in range(X.shape[0]):
            y_pred = X[tidx] @ ywhite[tidx]
            if saturate:
                for ridx in range(y_pred.shape[0]):
                    y_pred[ridx] = scaled_sigmoid(y_pred[ridx], lbound[ridx].item(), rbound[ridx].item(),
                                                         saturation_point=saturate_factor)

            genloss = loss_func(y_pred, y[tidx])
            generic_loss += genloss
        logging_dict['gen_loss'].append(generic_loss.detach().item())

        # Add regularizer considering spatial smoothness of bundles
        spatialreg_loss = torch.zeros(1)
        for tidx in range(X.shape[0]):
            spatialreg_loss += torch.linalg.norm(Ls @ ywhite[tidx])
        logging_dict['spatial_reg_loss'].append(spatialreg_loss.detach().item())

        # Add regularizer considering temporal smoothness of each bundles
        temporalreg_loss = torch.zeros(1)
        for tidx in range(X.shape[0]):
            temporalreg_loss += torch.linalg.norm(Lt @ ywhite[:, tidx])
        logging_dict['temporal_reg_loss'].append(temporalreg_loss.detach().item())

        sumloss = generic_loss + p1 * spatialreg_loss + p2 * temporalreg_loss

        # # Regularization on positive weights
        # if positive_constraint:
        #     negative_loss = torch.zeros(1)
        #     for tidx in range(X.shape[0]):
        #         negative_loss += torch.sum((ywhite[tidx] < 0) * torch.abs(ywhite[tidx]))
        #     logging_dict['positiveness_reg_loss'].append(negative_loss.detach().item())

        #     sumloss += negative_loss

        logging_dict['total_loss'].append(sumloss.detach().item())
        optimizer.zero_grad()
        sumloss.backward()
        optimizer.step()

    if verbose:
        print(f'Losses are decomposed into:')
        print(f'generic loss={generic_loss.detach()}')
        print(f'spatialloss={spatialreg_loss.detach()}')
        print(f'temporalloss={temporalreg_loss.detach()}')
        print(f'sumloss={sumloss.detach()}')
        # if positive_constraint:
        #     print(f'positivenessloss={negative_loss.detach()}')

    if positive_constraint:
        ywhite = (ywhite > 0) * ywhite
        
    pred_activity = ywhite.detach().numpy()

    if logging:
        return pred_activity, logging_dict
    else:
        return pred_activity

def interpolate_activity(fmri_bundles, bundles_ij, rcomb, dim, normalizing=True, wmmask=None, verbose=True, probaflag=False):
    """
    Interpolates fMRI activity along fiber bundles into a 3D volume.

    Parameters
    ----------
    fmri_bundles: List of 3D arrays, each array is a fMRI timeseries for a bundle
    bundles_ij: List of tuples, each tuple is a pair of connected bundles 
    rcomb: Combination weights for bundles
    dim: Dimensions of output volume
    normalizing: Whether to normalize voxel values 
    wmmask: Optional white matter mask
    verbose: Whether to print progress

    Returns
    -------
    wm_inpainted: 3D volume with interpolated fMRI signal
    wm_inpainted_masked: Masked version restricted to white matter
    """

    # Iterate across all the bundles and populate each voxels by the timcourses
    wm_inpainted = np.zeros(dim)
    wm_inpainted_masked = None
    normalizing_matrix = np.zeros(dim) # Count number of times a voxel belonged to a bundle to average out
    for k in tqdm(range(len(bundles_ij)), disable=not verbose):

        volcoords_interest = fmri_bundles[k]
        for coord in volcoords_interest:
            x,y,z,a = coord
            x,y,z = int(x),int(y),int(z)
            if probaflag:
                wm_inpainted[x,y,z] = wm_inpainted[x,y,z] + a * rcomb[k]
            else:
                wm_inpainted[x,y,z] = wm_inpainted[x,y,z] + rcomb[k]
            # normalizing_matrix[x,y,z] = normalizing_matrix[x,y,z] + np.abs(rcomb[k])
            normalizing_matrix[x,y,z] = normalizing_matrix[x,y,z] + 1.0

    normalizing_matrix[normalizing_matrix == 0] = np.inf
    if normalizing:
        wm_inpainted = wm_inpainted/np.abs(normalizing_matrix)    

    # Out of bound encoding for backgrounds
    wm_inpainted[wm_inpainted == 0] = -100000

    if not wmmask is None:
        # Apply a white matter mask if we care only about correct bundle masking
        wm_inpainted_masked = wm_inpainted * wmmask
        wm_inpainted_masked[wm_inpainted_masked == 0] = -100000
    
    return wm_inpainted, wm_inpainted_masked