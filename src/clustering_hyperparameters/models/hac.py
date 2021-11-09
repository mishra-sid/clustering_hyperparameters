from .clustering_model import ClusteringModel

from scipy.cluster.hierarchy import linkage, fclusterdata
@ClusteringModel.register('hac')
class HAC(ClusteringModel):
    def __init__(self, **parameters):
        super().__init__(False)
        self.params = parameters

    def fit(self, x):
        self.data = x

    def get_labels(self):
        return fclusterdata(self.data, **self.params)
