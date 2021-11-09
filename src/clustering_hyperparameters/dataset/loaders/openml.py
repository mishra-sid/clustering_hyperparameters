from clustering_hyperparameters.dataset.loaders.loader import DatasetLoader
from clustering_hyperparameters.dataset.dataset import Dataset

from sklearn.preprocessing import StandardScaler
from openml.datasets import get_dataset


@DatasetLoader.register("openml")
class OpenmlLoader(DatasetLoader):
    def __init__(self, name, metadata) -> None:
        super().__init__(name, metadata)

    def fetch_and_cache(self, cache_dir):
        dataset_id = self.metadata['id']
        dataset = get_dataset(dataset_id=dataset_id,
                                download_data=False)
        data = dataset.get_data(dataset_format='array')
        X, y = data[0][:, :-1], data[0][:, -1]
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        Dataset.store_from_data(name=self.name, 
                                cache_dir=cache_dir,
                                X=X,
                                y=y)
