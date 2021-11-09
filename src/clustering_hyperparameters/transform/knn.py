import numpy as np
from scipy.sparse import coo_matrix
import scipy
from scipy.spatial.distance import cdist

def sim_fn(XA,XB):
  return XA @ XB.T
  #return 1/ (1+ cdist(XA, XB))

def batched_knn(XA, XB, K, batch_size=1000, offset=0):
    K = np.minimum(K, XB.shape[0])
    res_i = np.zeros((XA.shape[0], K), dtype=np.int32)
    res = np.zeros((XA.shape[0], K), dtype=np.int32)
    resd = np.zeros((XA.shape[0], K), dtype=np.float32)
    for i in [x for x in range(0, XA.shape[0], batch_size)]:
        istart = i
        iend = min(XA.shape[0], i + batch_size)
        r = np.zeros((iend-istart, XB.shape[0]), dtype=np.float32)
        for j in range(0, XB.shape[0], batch_size):
            jstart = j
            jend = min(XB.shape[0], j + batch_size)
            r[:, jstart:jend] = sim_fn(XA[istart:iend], XB[jstart:jend])
        np.put(r, np.arange(iend - istart)*r.shape[1] + np.arange(istart, iend), np.inf)
        res[istart:iend, :] = np.argpartition(r, -K, axis=1)[:, -K:]
        resd[istart:iend, :] = r[np.arange(iend-istart)[:, None], res[istart:iend, :]]
        res_i[istart:iend, :] = np.repeat(np.expand_dims(np.arange(istart, iend), 1), K, axis=1) + offset

    row = res_i.flatten()
    col = res.flatten()
    d = resd.flatten()
    c = coo_matrix((d[row!=col], (row[row!=col], col[row!=col])), dtype=np.float32,shape=(XB.shape[0], XB.shape[0]))
    return c

def make_symmetric(coo_mat):
    lil = coo_mat.tolil()
    rows, cols = lil.nonzero()
    lil[cols, rows] = lil[rows, cols].maximum(lil[cols, rows])
    return lil.tocoo()

def knn_graph(vectors, k, batch_size, random_noise=0):
    graph = batched_knn(vectors, vectors, k,offset=0, batch_size=batch_size)
    graph.data += np.random.random(graph.data.shape) * random_noise
    graph = make_symmetric(graph)
    return graph 

