from clustering_hyperparameters.dataset.loaders.loader import DatasetLoader
from clustering_hyperparameters.dataset.dataset import Dataset

from sklearn.preprocessing import StandardScaler
from openml.datasets import get_dataset
from sentence_transformers import SentenceTransformer

import torch
from torchtext.datasets import DATASETS


@DatasetLoader.register("torchtext")
class TorchtextLoader(DatasetLoader):
    def __init__(self, name, metadata) -> None:
        super().__init__(name, metadata)
    
    def get_text_encoding(self, train_data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SentenceTransformer(self.metadata['encoder_model'], device=device)
        return model.encode(train_data, device=device)

    def fetch_and_cache(self, cache_dir):
        dataset = DATASETS[self.metadata['tag']](root=cache_dir + "/.root", split=self.metadata['split'])
        X = []
        y = []
        for label, line in dataset:
            X.append(line)
            y.append(label)
        X = self.get_text_encoding(X)
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        Dataset.store_from_data(name=self.name, 
                                cache_dir=cache_dir,
                                X=X,
                                y=y)