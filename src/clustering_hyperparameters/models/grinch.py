from .clustering_model import ClusteringModel

import grinch.grinch_alg as gr

import numpy as np

@ClusteringModel.register('grinch')
class Grinch(ClusteringModel):
    def __init__(self, **parameters):
        super().__init__(False)
        self.threshold = parameters.pop('threshold')

    def fit(self, x):
        self.model = gr.Grinch(x)
        self.model.build_dendogram()

    def get_labels(self):
        labels = self.model.flat_clustering(self.threshold)
        outliers = labels == -1
        labels[outliers] = np.arange(labels.shape[0], labels.shape[0] + np.count_nonzero(outliers))
        return labels
