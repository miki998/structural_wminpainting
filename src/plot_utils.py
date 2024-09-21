"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from src.utils import *

def plot_compare_inpainting(list_inpaintings: list, coords_of_interest: np.ndarray, 
                            affines: list, inpaint_names: list, vmin: int=-1, 
                            vmax: int=1, draw_cross: bool=False, thresh_subjapp: float=70):
    """
    Plot a comparison of multiple inpaintings by slicing through coordinates of interest.

    Parameters:
    -----------
    list_inpaintings (list of np.ndarrays): List of inpainting arrays to plot.
    coords_of_interest (np.ndarray): Coordinates to slice through for each inpainting.
    affines (list): Affine transforms for each inpainting image.  
    inpaint_names (list): Names of each inpainting to use in titles.
    vmin (int): Minimum value for colormap.
    vmax (int): Maximum value for colormap.  
    draw_cross (bool): Whether to draw crosshairs at slice coordinates.
    thresh_subjapp (float): Threshold of subjects bundle to show in title.

    Returns:
    --------
    None. Plots comparison of inpaintings.

    """
    r = len(list_inpaintings)
    nbc = 3
    if r % nbc == 0:
        nbr = r//nbc
    else:
        nbr = r//nbc + 1
    _, ax = plt.subplots(nbr,nbc, figsize=(8*nbc,5*nbr), facecolor='black')
    cmap = mpl.colormaps.get_cmap('jet')

    cmap.set_extremes(under='black', over='black')
    
    for r in range(nbr):
        for c in range(nbc):
            idx = r * nbc + c
            if nbr == 1: 
                plot_epi(nib.Nifti1Image(list_inpaintings[idx], affine=affines[idx]),
                         colorbar=True, cut_coords=coords_of_interest[idx], axes=ax[c],
                        cmap=cmap, vmin=vmin, vmax=vmax, draw_cross=draw_cross)
                ax[c].set_title(inpaint_names[idx], color='white')
            else:
                plot_epi(nib.Nifti1Image(list_inpaintings[idx], affine=affines[idx]),
                         colorbar=True, cut_coords=coords_of_interest[idx], axes=ax[r,c],
                        cmap=cmap, vmin=vmin, vmax=vmax, draw_cross=draw_cross)
                ax[r,c].set_title(inpaint_names[idx], color='white')

    # plt.suptitle(f'Bundle {int(np.ceil(thresh_subjapp))}% of subjects', color='white')