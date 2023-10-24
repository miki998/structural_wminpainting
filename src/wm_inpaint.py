from src.utils import *
from src.inpaint_utils import *
from src.fiberatlas_utils import *


def pipeline_inpainting(path2probatlas='../atlas_data/fiber_atlas/probconnatlas/wm.connatlas.scale1.h5', 
                        path2restdata='../atlas_data/rstfMRI_eg/rst_fmri_moviedata/',
                        region_of_interest=[], regularizer=0, norm='L2', 
                        weigthpath='./', voxelbundlepath='./',
                        inpaintpath='./'):
    """
    Full pipeline for inpainting from fibers structs and rest connectivity to fully inpainted connectivity bundles
    0: consistency-based
    1: length-based
    2: streamline number-based
    """
    # 1. Load the probabilistic atlas bundles
    print("1. Load the probabilistic atlas bundles ...")
    connFilename = path2probatlas
    hf = h5py.File(connFilename, 'r')

    centers = np.array(hf.get('header').get('gmcoords'))
    nsubject = hf.get('header').get('nsubjects')[()]
    gmregions_names = hf.get('header').get('gmregions')[()]
    bundle_affine = np.array(hf.get('header').get('affine'))[()]
    nb_regions = len(gmregions_names)
    
    # isolating region of interest that we want to compute the connectivity inpainting from
    index_of_interest = [np.where(gmregions_names.astype(str) == r)[0][0] 
                     for r in region_of_interest]

    
    # 2. Load the rest fmri in MNI space volumes
    # - Compute group rest timecourse at volume level in MNI
    print("2. Load the rest fmri in MNI space volumes ...")
    rest_runs = os.listdir(path2restdata)
    rst_vols = [nib.load(path2restdata+'{}'.format(run)) for run in rest_runs]

    rest_affine = rst_vols[0].affine
    ftimecourses = [rst.get_fdata() for rst in rst_vols]
    ftimecourses = np.asarray(ftimecourses)
    ftimecourse = ftimecourses.mean(axis=0)
    vdim = ftimecourse.shape[:3]
    rst_vol = nib.Nifti1Image(ftimecourse, affine=rest_affine)


    # 3. Extracting average timecourse and match voxel from probabilistic bundles with rest data
    print('3. Extracting average timecourse and match voxel from probabilistic bundles with rest data...')
    # - Transform the voxels to coordinates
    # - Match the coordinates i.e the voxels to a given label
    # - Obtain an mapping of a voxel to the parcels
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

    # 4. Compute FC of atlased timecourses
    print('4. Compute FC of atlased timecourses...')
    corr_mat = np.zeros((avg_tc.shape[0],avg_tc.shape[0]))
    for k in range(avg_tc.shape[0]):
        corr_mat[k] = nta.SeedCorrelationAnalyzer(avg_tc[k], avg_tc).corrcoef

    # 5. Compute the Bundle matrix that we use to regress
    print('5. Compute the Bundle matrix that we use to regress ...')
    thresh_subjapp = int(np.ceil(nsubject * 0.3)) # consider bundles that appaear at least in 30 % of the subjects    
    # Generating the X samples and the y samples
    # - Careful as well to remove the auto-correlation in the diagonal
    # - Raster scan parsing meaning that it is the activity of (R0,R1) -> (R0,R2) -> (R0,R3) etc...
    # Define matrix of end points on cortex
    X = []
    bundles_labels = []
    pairs = np.zeros((nb_regions, nb_regions))
    for i in tqdm(range(1,nb_regions + 1)):
        for j in range(i,nb_regions + 1):
            tmp = get_bundles_betweenreg(hf, i, j, verbose=False)
            if tmp is None: continue
            if np.sum(tmp[:,3] >= (thresh_subjapp)) == 0: continue
            bundles_labels.append((i,j))
            pairs[i-1,j-1] = 1.0
            vec = np.zeros(nb_regions)
            vec[i-1] = 1.0
            vec[j-1] = 1.0
            X.append(vec)
    X = np.asarray(X)
    y = corr_mat[index_of_interest].mean(axis=0)

    # 6. Regression of bundle weighting
    print('6. Regression of bundle weighting...')
    if regularizer == 0:
        consistency_view = get_aggprop(hf, 'consistency')
        scale = (np.array([consistency_view[p[0]-1,p[1]-1]/nsubject for p in bundles_labels])) ** 2
    elif regularizer == 1:
        length_view = get_aggprop(hf, 'length')
        scale = np.array([length_view[p[0]-1,p[1]-1] for p in bundles_labels])
        scale /= scale.max()
    else:
        nbStlines_view = get_aggprop(hf, 'numbStlines')
        scale = np.array([nbStlines_view[p[0]-1,p[1]-1] for p in bundles_labels])
        scale /= scale.max()

    lreg = optimize_lreg(X, y, scale, norm=norm, verbose=True)

    # Saving the weights if needed to re-use
    save(weigthpath + '/regressor.pkl', lreg)
    
    # 7. Populate the voxels with weighted connectivity
    print('7. Populate the voxels with weighted connectivity...')
    trans_affine = np.matmul(np.linalg.inv(rest_affine), bundle_affine)

    fmri_coords = []
    for k in tqdm(range(len(bundles_labels))):
        i,j = bundles_labels[k]
        streamline = get_bundles_betweenreg(hf, i,j)
        streamline = streamline[streamline[:,3] >= thresh_subjapp]

        # Transform voxel indexes of a volume to voxel index of another volume
        volcoords_interest = volcoord2mnicoord(streamline[:,[0,1,2]], trans_affine).astype(int)
        fmri_coords.append(volcoords_interest)
    # Save the space coords matching
    save(voxelbundlepath+'/bundlevox_coords.pkl', fmri_coords)

    # reconstruct with all bundles
    wm_inpainted_all, _ = interpolate_connectivity(fmri_coords, bundles_labels, lreg, corr_mat, vdim)

    t_index_bundle = []
    for k in range(len(bundles_labels)):
        a,b = bundles_labels[k]
        if (a in index_of_interest) or (b in index_of_interest):
            t_index_bundle.append(k)
    t_index_bundle = np.asarray(t_index_bundle)

    fmri_coords_t = [fmri_coords[t_index_bundle[k]] for k in range(len(t_index_bundle))]
    bundles_labels_t = [bundles_labels[t_index_bundle[k]] for k in range(len(t_index_bundle))]

    # Reconstruct only with bundles of interest intersecting with regions of interest
    wm_inpainted_rec, _ = interpolate_connectivity(fmri_coords_t, bundles_labels_t, lreg[t_index_bundle], corr_mat, vdim)

    save(inpaintpath+'/wm_inpainted_all.pkl', wm_inpainted_all)
    save(inpaintpath+'/wm_inpainted_rec.pkl', wm_inpainted_rec)

    # NOTE: Close the opened h5 file
    hf.close()