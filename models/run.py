import torch
import util
from pathlib import Path
import train


def main(args):
    # general
    if torch.cuda.is_available():
        device = torch.device("cuda")
        util.logging.info("Using CUDA")
    else:
        device = torch.device("cpu")
        util.logging.info("Using CPU")

    util.set_seed(args.seed)

    # init
    checkpoint_path = util.get_checkpoint_path(args)
    if not Path(checkpoint_path).exists():
        util.logging.info("Training from scratch")
        model, optimizer, stats = util.init(args, device)
    else:
        model, optimizer, stats, args = util.load_checkpoint(checkpoint_path, device)

    # train
    train.train(model, optimizer, stats, run_args=args)


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", default=0, type=int, help=" ")
    parser.add_argument("--num-hidden-units-1", default=10, type=int, help=" ")
    parser.add_argument("--num-hidden-units-2", default=10, type=int, help=" ")
    parser.add_argument("--num-iterations", default=100, type=int, help=" ")
    parser.add_argument("--log-interval", default=1, type=int, help=" ")
    parser.add_argument("--save-interval", default=10, type=int, help=" ")
    parser.add_argument("--checkpoint-interval", default=10, type=int, help=" ")

    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
