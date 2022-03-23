from os import name
from clustering_hyperparameters.dataset.dataset import Dataset
from .metrics import get_all_metrics
from clustering_hyperparameters.models.clustering_model import ClusteringModel


import numpy as np
from sklearn.preprocessing import normalize


from pathlib import Path
import time

def cluster(config):
    # setup model
    model, dataset, dataset_labels = setup(config)
    model.fit(dataset)
    predicted_labels = model.get_labels()
    metrics = get_all_metrics(dataset_labels, predicted_labels)
    return metrics


def setup(config):
    """
        Setup and return the datasets and model required for training.

        :param config: config dictionary
        :return: Tuple of datasets and model required for training.
    """
    dataset_index = int(config["dataset_index"])
    to_normalize = config["normalize"]
    dataset_name = config["suite"]["datasets"][dataset_index]["name"]

    suite_name = config["suite"]["name"]
    cache_dir = config["suite"]["cache_dir"]

    dataset_obj = Dataset(name=dataset_name, data_path=None, cache_dir=cache_dir, autoload=True)

    dataset, dataset_labels = dataset_obj.get_all()
    
    if to_normalize:
        dataset = normalize(dataset)    

    model_name = config["model"]["base_model"]
    model_params = config["model"]["params"]

    model = ClusteringModel.by_name(model_name)(**model_params)
    return model, dataset, dataset_labels
