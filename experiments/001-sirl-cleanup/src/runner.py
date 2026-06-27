import argparse
import yaml
from data_utils import load_data
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
    model = get_model(config, data)
    eval_model(config, data, model)


if __name__ == '__main__':
    args = parse_command_line_args()
    config = parse_config(args.config)
    main(config)