# Licensed under an MIT open source license - see LICENSE

"""

SCOUSE - Semi-automated multi-COmponent Universal Spectral-line fitting Engine
Copyright (c) 2016-2018 Jonathan D. Henshaw
CONTACT: henshaw@mpia.de

"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
import sys



def get_offsets(radius_pix):
    """
    Returns offsets of adjacent pixels

    Notes:

    For grid size of 3 - returns

    _offsets = np.array([[-1,1], [0,1], [1,1], [-1,0], [0,0], [1,0], [-1,-1], [0,-1], [1,-1]])

    etc.

    """
    arr = np.arange(radius_pix+1)
    sym = np.concatenate((arr,arr * -1)).astype(np.int)
    sym = np.unique(sym)

    _offsets = [pair for pair in itertools.product(sym,sym)]

    return _offsets

def neighbours(n_dim, idx, radius_pix):
    """
    Returns the indices of adjacent pixels
    """

    # Unravel the index of the selected spectrum
    unrav_idx = np.unravel_index(idx, np.flip(n_dim,0))

    # Get all the adjacent neighbours
    idxs = [tuple(c) for c in np.add(get_offsets(radius_pix), unrav_idx)]
    idxs = np.array(idxs)

    # Find out which of those neighbours are valid according to the shape of the
    # data cube
    validids = np.full(np.shape(idxs), np.nan)
    valid = (idxs[:,0] >= 0) & (idxs[:,0] < n_dim[1]) & (idxs[:,1] >= 0) & (idxs[:,1] < n_dim[0])
    validids[valid] = idxs[valid,:]

    # Package the valid neighburs up and send them back!
    indices_adjacent = [np.ravel_multi_index(np.array([int(validids[i,0]), int(validids[i,1])]), np.flip(n_dim, 0)) if np.isfinite(validids[i,0]) else np.nan for i in range(len(validids[:,0]))]

    return indices_adjacent
