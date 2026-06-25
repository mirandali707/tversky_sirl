import argparse

def parse_config():
    pass


def parse_command_line_args():
    parser = argparse.ArgumentParser("sirl-init")
    parser.add_argument("--config", help="Path to configuration file")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_command_line_args()
    print(args)