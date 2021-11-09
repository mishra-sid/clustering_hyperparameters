from .clustering_model import ClusteringModel

from sklearn.preprocessing import normalize
import hdbscan
import numpy as np

@ClusteringModel.register('hdbscan')
class HDBSCAN(ClusteringModel):
    def __init__(self, **parameters):
        super().__init__(False)
        self.model = hdbscan.HDBSCAN(**parameters)

    def fit(self, x):
        x = normalize(x)
        self.model.fit(x)

    def get_labels(self):
        labels = self.model.labels_
        outliers = labels == -1
        labels[outliers] = np.arange(labels.shape[0], labels.shape[0] + np.count_nonzero(outliers))
        return labels
