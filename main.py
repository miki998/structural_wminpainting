from src import *


if __name__ == '__main__':
    path2probatlas='../atlas_data/fiber_atlas/probconnatlas/wm.connatlas.scale1.h5'
   #  path2probatlas='../atlas_data/fiber_atlas/probconnatlas/wm.connatlas.scale3.h5'

    path2restdata='../atlas_data/rstfMRI_eg/rst_fmri_moviedata/'

    region_of_interest=['ctx-rh-posteriorcingulate', 'ctx-lh-posteriorcingulate']
    # region_of_interest = ['ctx-rh-lateraloccipital', 'ctx-lh-lateraloccipital']
    # region_of_interest = ['ctx-rh-inferiorparietal']
   #  region_of_interest = ['ctx-rh-posteriorcingulate_1', 'ctx-rh-posteriorcingulate_2',
   #     'ctx-lh-posteriorcingulate_1', 'ctx-lh-posteriorcingulate_2']

    normalizing=False
    regularizer=0 
    norm='L2'
    weigthpath='./'
    voxelbundlepath='./'
    inpaintpath='./'

    pipeline_inpainting(path2probatlas, 
                        path2restdata,
                        region_of_interest, normalizing, regularizer, norm, 
                        weigthpath, voxelbundlepath,
                        inpaintpath)