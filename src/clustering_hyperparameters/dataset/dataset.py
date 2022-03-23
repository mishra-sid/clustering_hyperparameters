import abc
import pathlib
import numpy as np


class Dataset:
    def __init__(self, name, data_path, cache_dir, autoload=False):
        self.name = name
        self.data_path = data_path
        self.cache_dir = cache_dir

        if autoload and not self.check_and_load_from_cache():
            self.load_data_from_path()
            self.preprocess()
            self.store_data_in_cache()
    
    @abc.abstractmethod
    def load_data_from_path(self):
        return

    @abc.abstractmethod
    def preprocess(self):
        return

    def check_and_load_from_cache(self):
        data_path = pathlib.Path(self.cache_dir) / (self.name + ".npz")

        if data_path.exists():
            with np.load(str(data_path)) as data:
                self.X = data['X']
                self.y = data['y']
            return True

        return False

    def store_data_in_cache(self):
        self.preprocess()
        pathlib.Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        cache_data_path = pathlib.Path(self.cache_dir) / self.name

        np.savez(str(cache_data_path), X=self.X, y=self.y)
    
    @classmethod
    def store_from_data(cls, name, cache_dir, X, y):
        dataset = cls(name=name, data_path=None, cache_dir=cache_dir)
        dataset.X = X
        dataset.y = y
        dataset.store_data_in_cache()
    
    def get_all(self):
        return self.X, self.y

