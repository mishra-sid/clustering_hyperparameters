from .clustering_model import ClusteringModel

import sklearn.cluster as skcluster
from clustering_hyperparameters.transform.knn import knn_graph

@ClusteringModel.register('spectral')
class Spectral(ClusteringModel):
    def __init__(self, **parameters):
        super().__init__(False)
        self.model = skcluster.SpectralClustering(**parameters)

    def fit(self, x):
        self.model.fit(x)

    def get_labels(self):
        return self.model.labels_
