from .clustering_model import ClusteringModel
from clustering_hyperparameters.transform.knn import knn_graph

from scc.scc import Affinity

@ClusteringModel.register('affinity')
class Affinity(ClusteringModel):
    def __init__(self, **parameters):
        super().__init__(False)
        self.n_neighbours = parameters.pop('n_neighbors')
        self.batch_size = parameters.pop('batch_size')
        self.num_rounds = parameters.pop('num_rounds')

    def fit(self, x):
        x_knn_graph = knn_graph(x)
        self.model = Affinity(g=x_knn_graph,num_rounds=self.num_rounds)
        self.model.fit()

    def get_labels(self):
        return self.model.rounds[self.num_rounds - 1].cluster_assignments
