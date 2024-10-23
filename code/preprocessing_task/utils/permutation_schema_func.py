import numpy as np

def permutation_schema(n_tpoints, n_perms=1000, chunk_size=15, seed=0, flip=True):
    
    """
    Creates an index permutation schema
    
    Inputs:
    - n_tpoints : int, number of timepoints (rows)
    - n_perms : int, number of permutations (columns); default = 1000
    - chunk_size: int, size of chunks to be kept contiguous, in TR; default = 15 (30s)
    - seed: int, seed for the random permutation; default = 0
    - flip: bool, decide wether to flip or not random number of chunks, default = True

    Outputs:
    - perm_schema : array, 2d matrix of shape = n_tpoints by n_perms + 1; first row contains unshuffled indices
    """

    # Set seed
    np.random.seed(seed)
    
    # Create an array of indices from 0 to number of timepoints
    indices = np.arange(n_tpoints)

    # Initialize the permutation schema matrix
    perm_schema = np.zeros((n_tpoints, n_perms+1), dtype=int)

    # Create chunks of contiguous timepoints
    chunks = [indices[i:i + chunk_size] for i in range(0, n_tpoints, chunk_size)]

    # Flip some of the chunks
    if flip:
        flip_inds = np.random.choice([0, 1], size=len(chunks))
        chunks = [np.flip(chunk) if flip_inds[c] == 1 else chunk for c, chunk in enumerate(chunks)]

    # Shuffle the order of the chunks separately for each permutation
    for i in range(1,n_perms+1):
        perm_schema[:,i] = np.concatenate([list(chunks[c]) for c in np.random.permutation(len(chunks))])
        
    # Assign original indices to first column of permutation schema
    perm_schema[:,0] = indices

    return perm_schema