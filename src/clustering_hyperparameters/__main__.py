from .optimize.optimize import optimize
from .dataset.suite import DatasetSuite
import hydra
import math


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    """ Parses the hydra config, fetches the datasets present in the suite and starts optimization trials

    Args:
        cfg ([type]): [description]
    """
    cfg.dataset_index = str(cfg.dataset_index)
    
    if cfg.fetch_suite:
        DatasetSuite.fetch_and_cache_from_cfg(cfg.suite, cfg.dataset_index)
    
    optimize(cfg)
