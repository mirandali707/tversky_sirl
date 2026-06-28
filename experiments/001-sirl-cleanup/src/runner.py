import argparse
import yaml
import json
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
    for seed in config["seeds"]:
        row = {
            "seed": seed
        }
        set_all_seeds(seed)

        model, ckpt_path = get_model(config, data, results_dir)
        row["ckpt_path"] = ckpt_path

        results = eval_model(config, data, model)
        row = row | results # add results to row
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_path = results_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"saved results to {results_path}")


if __name__ == '__main__':
    args = parse_command_line_args()
    config = parse_config(args.config)
    main(config)