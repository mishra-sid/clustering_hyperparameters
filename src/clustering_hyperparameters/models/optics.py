from .clustering_model import ClusteringModel
import sklearn.cluster as skcluster
import numpy as np

@ClusteringModel.register('optics')
class OPTICS(ClusteringModel):
    def __init__(self, **parameters):
        super().__init__(False)
        self.model = skcluster.OPTICS(**parameters)

    def fit(self, x):
        self.model.fit(x)

    def get_labels(self):
        labels = self.model.labels_
        outliers = labels == -1
        labels[outliers] = np.arange(labels.shape[0], labels.shape[0] + np.count_nonzero(outliers))
        return labels
