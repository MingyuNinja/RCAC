import argparse
import os
from research.utils.trainer import Config, train

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--path", "-p", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--env", "-e", type=str, default=None)
    parser.add_argument("--seed", "-s", type=int, default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.env is not None: config['env'] = args.env
    if args.env is not None: config['seed'] = args.seed
    train(config, args.path, device=args.device)