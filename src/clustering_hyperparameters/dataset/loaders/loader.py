from registrable import Registrable

import abc

class DatasetLoader(Registrable):
    def __init__(self, name, metadata) -> None:
        self.name = name
        self.metadata = metadata
    
    @abc.abstractmethod
    def fetch_and_cache(self, cache_dir):
        return
