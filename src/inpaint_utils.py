from src.utils import *
from src.fiberatlas_utils import *


def property_regularizer(x, C, norm='L2'):
    """
    Compute regularizer dependent on bundle property following:
    - Lower consistency increase penalization
    - Lower length increase penalization
    - Lower number of streamlines increase penalization

    C vector's entries range in [0,1]
    """
    inv_C = (1 / C)
    if norm == 'L2':
        ret = ((inv_C * x) ** 2).sum() / C.shape[0]
    elif norm == 'L1':
        ret = ((inv_C * x).abs()).sum() / C.shape[0]
    elif norm == 'PL2':
        # inv_C += 0.1 # buffer infinite
        ret = (x.abs()**inv_C).sum() / C.shape[0]
    else:
        print('Unsupported Norm')
    return ret

def optimize_lreg(X, y, prop, norm, num_epochs=1000, lr=1e-3, seed=99, verbose=False):
    """
    Optimize linear regression parameters (MLP 1 layer) with property based regularizer
    """
    torch.manual_seed(seed)
    # Model defining
    lreg = nn.Sequential(
        nn.Linear(len(X), 1),
    )

    # Use L1 loss for sparsity of weights
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(lreg.parameters(), lr=lr)

    Xtorch = torch.Tensor(X.T)
    ytorch = torch.Tensor(y[:,None])
    for n in tqdm(range(num_epochs)):
        y_pred = lreg(Xtorch)

        # Add regularizer considering consistency of bundles
        reg_loss = property_regularizer(lreg[0].weight[0], torch.Tensor(prop), norm=norm)
        genloss = loss_func(y_pred, ytorch)
        loss =  genloss + reg_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if verbose:
        print(f'Losses are decomposed into generic loss={genloss.detach()} and regularizer loss={reg_loss.detach()}')
    res_weight = lreg[0].weight.detach().numpy()[0]
    return res_weight


def interpolate_connectivity(fmri_bundles, bundles_ij, rcomb, regions_correlation, dim, normalizing=True, wmmask=None):
    """
    Compute interpolated connectivity on white matter bundles
    """
    # Iterate across all the bundles and populate each voxels by the timcourses
    wm_inpainted = np.zeros(dim)
    wm_inpainted_masked = None
    normalizing_matrix = np.zeros(dim) # Count number of times a voxel belonged to a bundle to average out
    for k in tqdm(range(len(bundles_ij))):
        i,j = bundles_ij[k]

        volcoords_interest = fmri_bundles[k]
        avg_betweenbundle_conn = regions_correlation[i-1,j-1]

        for coord in volcoords_interest:
            x,y,z = coord
            wm_inpainted[x,y,z] = wm_inpainted[x,y,z] + rcomb[k] *  avg_betweenbundle_conn
            normalizing_matrix[x,y,z] = normalizing_matrix[x,y,z] + np.abs(rcomb[k])

    normalizing_matrix[normalizing_matrix == 0] = np.inf
    if normalizing:
        wm_inpainted = wm_inpainted/np.abs(normalizing_matrix)    

    # Out of bound encoding for backgrounds
    wm_inpainted[wm_inpainted == 0] = -10

    if not wmmask is None:
        # Apply a white matter mask if we care only about correct bundle masking
        wm_inpainted_masked = wm_inpainted * wmmask
        wm_inpainted_masked[wm_inpainted_masked == 0] = -10
    
    return wm_inpainted, wm_inpainted_masked