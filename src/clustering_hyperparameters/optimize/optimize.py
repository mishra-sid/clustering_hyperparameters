from ..cluster.cluster import cluster
from ..utils.type_utils import get_type_from_str

from pathlib import Path

from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax.core.metric import Metric
from ax.service.utils.report_utils import exp_to_df

from ray import tune
from ray.tune import report, Callback
from ray.tune.suggest.ax import AxSearch
from omegaconf import OmegaConf
import torch
import pandas as pd


def evaluation_metric_function(parameterization, config):
    """[Finds evaluation metric for a model given it's parameters and global config]

    Args:
        parameterization (dict): [Parameter values taken by each hyperparameter of model]
        config (dict): [Global config object]
    """
    resolved_config = config
    resolved_config["model"]["params"] = dict(parameterization)
    
    report(**cluster(resolved_config))

def optimize(config):
    """[Performs hyperparameter optimization procedure given global config object]

    Args:
        config (dict): [Global config object]
    """
    dataset_index = int(config['dataset_index'])
    suite_name = config['suite']['name']
    output_dir = config["root_dir"] + "/output/" + suite_name
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    optim_seed = config["optim"]["run_index"]
    normalized = "_normalized" if config["normalize"] else ''
    exp_name = f"experiment_{config['model']['name']}_{config['suite']['datasets'][dataset_index]['name']}{normalized}_{optim_seed}"
    
    num_random_trials = config["optim"]["num_random_trials"]
    to_attach = config["optim"]["attach_from_existing_trials"]
    num_bayes_trials = config["optim"]["num_bayes_trials"]
    num_total_trials = num_random_trials + num_bayes_trials
    gen_steps = []
    if num_random_trials > 0:
        gen_steps.append(GenerationStep(
                model=Models.SOBOL,
                num_trials=num_random_trials,
                min_trials_observed=num_random_trials,
                max_parallelism=config["optim"]["compute"]["max_concurrent"],
                model_kwargs={"seed": optim_seed}))
    
    if num_bayes_trials > 0:
         gen_steps.append(GenerationStep(
                model=Models.BOTORCH,
                num_trials=num_bayes_trials,
                max_parallelism=config["optim"]["compute"]["max_concurrent"]))
    
    gen_strat = GenerationStrategy(
        steps=gen_steps
    )
    
    ax_client = AxClient(generation_strategy=gen_strat,
                         enforce_sequential_optimization=False)
    
    model_params = OmegaConf.to_container((config["model"]["params"]), resolve=True)
    ax_client.create_experiment(name="clustering_hyperparameter_optimization",
                                parameters=model_params,
                                objective_name=config["optim"]["eval_metric"]["name"],
                                minimize=config["optim"]["eval_metric"]["minimize"])

    ax_client.experiment.add_tracking_metrics([Metric(name=mname) for mname in ["adjusted_mutual_info_score",
                                                                                "completeness_score",
                                                                                "fowlkes_mallows_score",
                                                                                "homogeneity_score",
                                                                                "mutual_info_score",
                                                                                "normalized_mutual_info_score",
                                                                                "rand_score",
                                                                                "v_measure_score",
                                                                                "homogeneity_completeness_v_measure"]])
    out_csv_path = Path(output_dir) / (exp_name + ".csv" )
    if to_attach:
       out_bayes_path = Path(output_dir) / (exp_name + "_bayesian.csv" ) 
       
       if out_bayes_path.exists():
           return

       num_trials_to_attach = config["optim"]["num_trials_to_attach"]
       trials_df_existing = pd.read_csv(out_csv_path)
       for ind, row in trials_df_existing.head(num_trials_to_attach).iterrows():
           param_cols=[x for x in trials_df_existing.columns if x not in ["adjusted_rand_score", 
                                                                          "trial_status",
                                                                          "generator_model",
                                                                          "generation_method",
                                                                          "trial_index",
                                                                          "arm_name",
                                                                          "compute_time"]]
           get_type = lambda x: list(filter(lambda y: y['name'] == x, model_params))[0]['value_type']
           params = { col: get_type_from_str(get_type(col))(row[col]) for col in param_cols }
           params, trial_ind = ax_client.attach_trial(params)
           ax_client.complete_trial(trial_index=trial_ind, 
                                    raw_data={"adjusted_rand_score": row["adjusted_rand_score"]})

    elif out_csv_path.exists():
        return

   
    resolved_config = OmegaConf.to_container(config, resolve=True)
    
    class TuneCallBack(Callback):
        def on_step_begin(self, iteration, trials, **info):
            torch.manual_seed(optim_seed)

    tune.run(
        lambda parameterization: evaluation_metric_function(parameterization, resolved_config),
        name=exp_name,
        num_samples=num_total_trials,
        callbacks=[TuneCallBack()],
        search_alg=AxSearch(
            ax_client=ax_client,
            max_concurrency=config["optim"]["compute"]["max_concurrent"]
        ),
        local_dir=config["root_dir"] + "/ray/" + suite_name,
        resources_per_trial={"cpu": config["optim"]["compute"]["cpu"], "gpu": config["optim"]["compute"]["gpu"]}
    )

    trials_df = ax_client.get_trials_data_frame()
    compute_time_col = { index: (trial.time_completed - trial.time_run_started).total_seconds() for index, trial in ax_client.experiment.trials.items() }
    trials_df['compute_time'] = [ compute_time_col[trial_index] for trial_index in trials_df.trial_index ]

    trials_df.to_csv(output_dir + "/" + exp_name + ".csv", encoding='utf-8', index=False)

