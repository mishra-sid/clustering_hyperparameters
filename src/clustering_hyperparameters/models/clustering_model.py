from registrable import Registrable

import abc

class ClusteringModel(Registrable):
    """Abstract Clustering Model Object.
       Extend this class and override fit and get_labels method to add a new method.
       Register the model by adding @ClusteringModel.register('model_name') and import the new method in models/__init__.py
    """
    def __init__(self,
                 mini_batch=False):
        self.mini_batch = mini_batch
        self.model = None

    @abc.abstractmethod
    def fit(self, x):
        return

    @abc.abstractmethod
    def get_labels(self):
        return
