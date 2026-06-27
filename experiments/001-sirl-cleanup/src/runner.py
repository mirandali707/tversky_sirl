import argparse
from types import SimpleNamespace

import yaml

from data_utils import load_data


# def _to_namespace(obj):
#     """Recursively convert dicts to SimpleNamespace for dotted attribute access."""
#     if isinstance(obj, dict):
#         return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
#     if isinstance(obj, list):
#         return [_to_namespace(v) for v in obj]
#     return obj


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
    return data


if __name__ == '__main__':
    args = parse_command_line_args()
    config = parse_config(args.config)
    main(config)