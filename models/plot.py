import util
import matplotlib.pyplot as plt
from pathlib import Path
import torch


def plot_losses(path, losses):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
    ax.plot(losses)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    fig.tight_layout(pad=0)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    util.logging.info(f"Saved to {path}")
    plt.close(fig)


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        util.logging.info("Using CUDA")
    else:
        device = torch.device("cpu")
        util.logging.info("Using CPU")

    if args.checkpoint_path is None:
        checkpoint_paths = list(util.get_checkpoint_paths())
    else:
        checkpoint_paths = [args.checkpoint_path]

    for checkpoint_path in checkpoint_paths:
        try:
            model, optimizer, stats, run_args = util.load_checkpoint(checkpoint_path, device=device)
        except FileNotFoundError as e:
            print(e)
            if "No such file or directory" in str(e):
                print(e)
                continue

        plot_losses(f"{util.get_save_dir(run_args)}/losses.pdf", stats.losses)

        # ... plot other stuff with `model` ...


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--repeat", action="store_true", help="")
    parser.add_argument("--checkpoint-path", type=str, default=None, help=" ")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    with torch.no_grad():
        if args.repeat:
            while True:
                main(args)
        else:
            main(args)
