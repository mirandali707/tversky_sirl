import argparse
import yaml
from data_utils import load_data

def parse_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_command_line_args():
    parser = argparse.ArgumentParser("sirl-init")
    parser.add_argument("--config", default="configs/sirl.yaml",
                        help="Path to configuration file")
    return parser.parse_args()


def main(config):
    anchors, positives, negatives = load_data(config)
    print("loaded")
    print(anchors.shape)


if __name__ == '__main__':
    args = parse_command_line_args()
    config = parse_config(args.config)
    main(config)