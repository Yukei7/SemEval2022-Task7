import json
import torch
import numpy as np
import logging
from train import train
from utils import print_header


def main(args):

    logging.basicConfig(level=logging.DEBUG)

    # gpu setting
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # seed setting
    torch.manual_seed(args["seed"])
    if use_cuda:
        torch.cuda.manual_seed_all(args["seed"])
        torch.backends.cudnn.benchmark = True
    np.random.seed(args["seed"])

    logging.debug(f"Device: {str(device)}; Seed: {args['seed']}")

    ##############
    phase = args["phase"]

    assert phase in ["train", "test"]
    logging.debug(f"Subtask: {args['subtask']}, {phase} phase")

    if phase == "train":
        train(args, device)
    else:
        pass


if __name__ == "__main__":
    with open("config.json", "r") as f:
        conf = json.load(f)
    main(conf)
