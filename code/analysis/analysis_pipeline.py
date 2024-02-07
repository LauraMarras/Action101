import numpy as np
from canonical_correlation_funcs import run_cca_all_subjects

if __name__ == '__main__': 

    print('Starting CCA')

    # Set parameters
    sub_list = np.array([7,8,9])
    n_subs = len(sub_list)
    save = True

    ## CCA
    atlas_file = 'atlas_2orig' # 'atlas1000_2orig.nii.gz'
    pooln = 25

    ## Permutation schema
    n_perms = 1000
    chunk_size = 15
    seed = 0

    # Run CCA for all subjects
    run_cca_all_subjects(sub_list, atlas_file, n_perms, chunk_size, seed, pooln, save)

    print('Finished CCA')