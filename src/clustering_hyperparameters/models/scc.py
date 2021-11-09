from .clustering_model import ClusteringModel
from clustering_hyperparameters.transform.knn import knn_graph

from scc.scc import Affinity
import numpy as np

@ClusteringModel.register('scc')
class SCC(ClusteringModel):
    def __init__(self, **parameters):
        super().__init__(False)
        self.n_neighbours = parameters.pop('n_neighbors')
        self.batch_size = parameters.pop('batch_size')
        self.num_rounds = parameters.pop('num_rounds')
        tau_lower, tau_upper = parameters.pop('tau_lower'), parameters.pop('tau_upper')
        self.taus = np.geomspace(start=tau_upper, stop=tau_lower, num=self.num_rounds)
        

    def fit(self, x):
        x_knn_graph = knn_graph(x)
        self.model = SCC(g=x_knn_graph,num_rounds=self.num_rounds, taus=self.taus)
        self.model.fit(x_knn_graph)

    def get_labels(self):
        return self.model.rounds[self.num_rounds - 1].cluster_assignments
