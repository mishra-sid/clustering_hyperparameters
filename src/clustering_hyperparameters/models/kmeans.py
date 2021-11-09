import numpy as np
from .clustering_model import ClusteringModel
import sklearn.cluster as skcluster
import faiss

@ClusteringModel.register('kmeans')
class KMeans(ClusteringModel):
    def __init__(self, **parameters):
        super().__init__(False)
        self.model = skcluster.KMeans(**parameters)

    def fit(self, x):
        self.model.fit(x)

    def get_labels(self):
        return self.model.labels_

@ClusteringModel.register('kmeans-minibatch')
class MiniBatchKMeans(ClusteringModel):
    def __init__(self, **parameters):
        super().__init__(True)
        self.model = skcluster.MiniBatchKMeans(**parameters)

    def fit(self, x):
        self.model.fit(x)

    def get_labels(self):
        return self.model.labels_

@ClusteringModel.register("kmeans-faiss")
class FaissKMeans(ClusteringModel):
    def __init__(self, **params):
        super().__init__(False)
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.params = params

    def fit(self, X):
        self.X = np.ascontiguousarray(X)
        self.kmeans = faiss.Kmeans(d=self.X.shape[1],
                                   **self.params)
        self.kmeans.train(self.X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def get_labels(self):
        return self.kmeans.index.search(self.X.astype(np.float32), 1)[1].flatten()
