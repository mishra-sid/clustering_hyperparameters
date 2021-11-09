import higra as hg
import numpy as np
import scipy

def get_coo_sims(x):
    """ Get cooccurence matrix given data

    Args:
        x (ndarray): Input data

    Returns:
        [ndarray]: coocurrence matrix
    """
    z = (x * x).sum(1, keepdims=True) ** 0.5
    return scipy.sparse.coo_matrix(x @ x.T / z / z.T)

def coo_2_hg(coo_mat):
    """ Converts cooccurrence matrix to higra graph

    Args:
        coo_mat (ndarray): Input cooccurence matrix

    Returns:
        [type]: output higra graph
    """
    lil = coo_mat.tolil()
    lil[np.arange(lil.shape[1] - 1), np.arange(lil.shape[1] - 1) + 1] = 1e-9
    coo_mat = lil.tocoo()
    rows = coo_mat.row[coo_mat.row < coo_mat.col]
    cols = coo_mat.col[coo_mat.row < coo_mat.col]
    dists = coo_mat.data[coo_mat.row < coo_mat.col]
    ugraph = hg.higram.UndirectedGraph(coo_mat.shape[0])
    ugraph.add_edges(rows, cols)
    return ugraph, dists.astype(np.float32)
