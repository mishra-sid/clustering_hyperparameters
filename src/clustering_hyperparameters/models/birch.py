from .clustering_model import ClusteringModel


import sklearn.cluster as skcluster


@ClusteringModel.register('birch')
class BIRCH(ClusteringModel):
    def __init__(self, **parameters):
        super().__init__(False)
        self.model = skcluster.Birch(**parameters)

    def fit(self, x):
        self.model.fit(x)

    def get_labels(self):
        return self.model.labels_
