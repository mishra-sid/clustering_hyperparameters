# Clustering Hyperparameters

This code provides a benchmarking framework to evaluate ease of tuning of hyperparameters in a clustering algorithm.

## Installations Instructions
1.  To create a virtual environment using conda, with all the dependencies installed use:
    ```bash
        conda env create --name clustering-env --file=environment.yml
    ```
    Optionally if you prefer pip, create a virtual environment and run:
    ```bash
        pip install -r requirements.txt
    ```

2. After cloning the repo, you can install all required dependencies by running `make install` in the root directory. (Note: this will install python modules, so be sure you are in the proper virtual environment.)

3. **[Optional]** To build a Docker environment use the provided `Dockerfile` or `docker-compose.yml`, go to the root directory of the repository and perform:
   ```bash
      # Using docker build
      docker build -t clustering-benchmark .

      # Using docker compose
      docker-compose up
   ```
## Usage Instruction

This module provides a command line interface powered by [Facebook Hydra](https://hydra.cc/) available with `clustering_hyperparameters` command. 
The default options are provided in [Global Config file](src/clustering_hyperparameters/conf/config.yaml) 

They can be overriden using the cli in the following way:
```bash
    clustering_hyperparameters override_key1=override_value1
                               override_key2=override_value2 ... 
                               override_key_n=override_value_n

    # e.g
    clustering_hyperparameters suite=nlp model=dbscan
```

## Extending the framework

#### Modify hyperparameter ranges
Modify the config file at `src/clustering_hyperparameters/conf/model/<model>.yaml` e.g. [dbscan model config file](src/clustering_hyperparameters/conf/model/dbscan.yaml)

You can use dynamic ranges using hydra's variable interpolation e.g. [kmeans-minibatch config file](src/clustering_hyperparameters/conf/model/kmeans-minibatch.yaml)
Modified dbscan parameter ranges
```yaml
    name: dbscan
    params:
    - name: metric
        type: fixed
        value_type: str
        value: cosine
    # Updated eps upper bound from 0.999 to 1.999
    - name: eps
        type: range
        value_type: float
        bounds: [0.001, 1.999]
    # Updated min_samples upper bound from 100 to 500
    - name: min_samples
        type: range
        value_type: int
        bounds: [2, 500]

```
#### Add a new Model
Add a new file to [model dir](src/clustering_hyperparameters/models/)
```python
    @ClusteringModel.register('my-new-model')
    class MyNewModel(ClusteringModel):
        def __init__(self, **parameters):
            # Initialize parameters
        
        def fit(self, x):
            # Define how to fit a model

        def get_labels(self):
            # Define how to get clustering label assignments
```
To use the new model, run:
```bash
    clustering_hyperparameters model=my-new-model
```

#### Define a new dataset collection/suite
To define a new suite, create a new file in [suite dir](src/clustering_hyperparameters/conf/suite/) in the following format:

```yaml
    name: my-suite
    cache_dir: ${root_dir}/data/my-suite
    datasets:
    - name: mfeat-fourier
      loader: openml
      metadata:
        id: 14
        num_instances: 2000
        num_features: 76
        num_clusters: 10
    ....

    - name: AGNews-paraphrase-mpnet
      loader: torchtext
      metadata:
        tag: AG_NEWS
        split: test
        encoder: sentence-transformer
        encoder_model: paraphrase-mpnet-base-v2
        num_instances: 7600
        num_features: 768   
        num_clusters: 4
    
```

#### Define a new dataloader
```python
    @DatasetLoader.register("my-data-loader")
    class MyDataLoader(DatasetLoader):
        def __init__(self, name, metadata)
            # Initialize metadata

        def fetch_and_cache(self, cache_dir):
            # Fetch dataset, perform preprocessing and cache it using `Dataset.store_from_data` utility
```

To use this data loader, use `loader: my-new-loader` in suite config file.
## Results
The notebooks with the results/plots found in the paper can be found as jupyter notebook in [experiments](experiments) directory. It contains plots for fANOVA analysis, $EoM_{R}$ / $EoM_B$ for generic and nlp datasets.

## Reproducibility
#### Obtain Evaluations provided in paper
In a SLURM environment, run the script file:
```bash
   ./bin/run_all_exps.sh
```
This will spawn multiple sbatch jobs which will run all the required evaluations in parallel.

#### Reproduce results/plots provided in paper using provided evaluations 

For ease of tweaking, the evaluated outputs described in the paper are provided in [output](output.zip)

To reproduce the results:
1. Extract `output.zip` to `output/`
2. Go to any jupyter notebook in [experiments](experiments) folder and run the notebook to get the corresponding plots.
