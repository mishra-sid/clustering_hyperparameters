from .loaders.loader import DatasetLoader

from omegaconf import OmegaConf

class DatasetSuite:
    def __init__(self, name, cache_dir, datasets):
        self.name = name
        self.cache_dir = cache_dir
        self.datasets = datasets

    def fetch_and_cache_dataset(self, dataset_index):
        loader_type = self.datasets[int(dataset_index)].pop('loader')
        loader_cls = DatasetLoader.by_name(loader_type)
        loader = loader_cls(**self.datasets[int(dataset_index)])
        loader.fetch_and_cache(self.cache_dir)
    
    @classmethod
    def fetch_and_cache_from_cfg(cls, cfg, dataset_index):
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        suite = cls(resolved_cfg['name'], resolved_cfg['cache_dir'], resolved_cfg['datasets'])
        suite.fetch_and_cache_dataset(dataset_index)
