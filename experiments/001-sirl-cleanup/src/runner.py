import argparse
import yaml
import json
import itertools
import pandas as pd
from pathlib import Path
from utils import *
from models import get_model
from eval import eval_model

def parse_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_command_line_args():
    parser = argparse.ArgumentParser("sirl-init")
    parser.add_argument("--config", default="configs/sirl.yaml",
                        help="Path to configuration file")
    return parser.parse_args()


def get_all_model_configs(config):
    """
    enumerates all possible combinations of parameter values from the "model" section of the config.
    for example, if we have
    { ...
        latent_dim: [3,5],
        fbank_size: [8,16]  
    ...}
    this will return the list of dicts 
    [
    {"latent_dim": 3, "fbank_size: 8},
    {"latent_dim": 5, "fbank_size: 8},
    {"latent_dim": 3, "fbank_size: 16},
    {"latent_dim": 5, "fbank_size: 16},
    ]

    only params whose value is a list are treated as varying; everything else
    is held fixed and left out of the returned dicts. if no params are lists,
    returns [{}] so the caller still runs exactly one configuration.
    """
    model_params = config["model"]
    list_keys = [k for k, v in model_params.items() if isinstance(v, list)]
    if not list_keys:
        return [{}]
    list_values = [model_params[k] for k in list_keys]
    return [dict(zip(list_keys, combo))
            for combo in itertools.product(*list_values)]

def main(config):
    data = load_data(config)

    results_dir = Path("results") / config["experiment_name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # save expt metadata in results dir
    metadata_path = results_dir / "metadata.json"
    metadata = {
        "config": config
    }
    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)

    rows = []
    results_path = results_dir / "results.csv"
    for param_permutation in get_all_model_configs(config):
        # overwrite the varying params inside config["model"] (where get_model reads them),
        curr_config = config | {"model": config["model"] | param_permutation}
        for seed in config["seeds"]:
            row = {
                "seed": seed,
            } | param_permutation # add this permutation of listed params to the row
            set_all_seeds(seed)

            model, ckpt_path = get_model(curr_config, data, results_dir, seed)
            row["ckpt_path"] = ckpt_path

            results = eval_model(curr_config, data, model)
            row = row | results # add results to row
            rows.append(row)
            results_df = pd.DataFrame(rows)
            results_df.to_csv(results_path, index=False)
    print(f"saved results to {results_path}")


if __name__ == '__main__':
    args = parse_command_line_args()
    config = parse_config(args.config)
    main(config)