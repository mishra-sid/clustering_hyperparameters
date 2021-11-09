from .clustering_model import ClusteringModel
import scipy.cluster as scc
import higra as hg

from clustering_hyperparameters.utils.similarity import coo_2_hg, get_coo_sims
from clustering_hyperparameters.transform.knn import knn_graph

@ClusteringModel.register('hac_exp')
class HAC_Exp(ClusteringModel):
    def __init__(self, **parameters):
        super().__init__(False)
        self.alpha = parameters.pop('alpha')
        self.n_neighbours = parameters.pop('n_neighbors')
        self.batch_size = parameters.pop('batch_size')
        self.params = parameters

    def get_linkage_matrix(self, coo_sims):
        ugraph, edge_weights = coo_2_hg(coo_sims)
        edge_weights = 1 / (1 + edge_weights)
        tree, altitudes = hg.binary_partition_tree_exponential_linkage(ugraph, edge_weights, self.alpha)
        Z = hg.binary_hierarchy_to_scipy_linkage_matrix(tree, altitudes=altitudes)
        return Z

    def fit(self, x):
        # dataset_coo_sims = knn_graph(x, self.n_neighbours, self.batch_size)
        dataset_coo_sims = get_coo_sims(x)
        self.Z = self.get_linkage_matrix(dataset_coo_sims)

    def get_labels(self):
        return scc.hierarchy.fcluster(self.Z, **self.params)
