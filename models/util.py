import collections
import torch
import models
from pathlib import Path
import logging
import sys
import numpy as np
import os


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(pathname)s:%(lineno)d | %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


def get_path_base_from_args(args):
    return f"h1={args.num_hidden_units_1}_h2={args.num_hidden_units_2}_seed={args.seed}"


def get_save_job_name_from_args(args):
    return get_path_base_from_args(args)


def get_save_dir_from_path_base(path_base):
    return f"save/{path_base}"


def get_save_dir(args):
    return get_save_dir_from_path_base(get_path_base_from_args(args))


def get_checkpoint_path(args, checkpoint_iteration=-1):
    return get_checkpoint_path_from_path_base(get_path_base_from_args(args), checkpoint_iteration)


def get_checkpoint_path_from_path_base(path_base, checkpoint_iteration=-1):
    checkpoints_dir = f"{get_save_dir_from_path_base(path_base)}/checkpoints"
    if checkpoint_iteration == -1:
        return f"{checkpoints_dir}/latest.pt"
    else:
        return f"{checkpoints_dir}/{checkpoint_iteration}.pt"


def get_checkpoint_paths(checkpoint_iteration=-1):
    save_dir = "./save/"
    for path_base in sorted(os.listdir(save_dir)):
        yield get_checkpoint_path_from_path_base(path_base, checkpoint_iteration)


def init(run_args, device):
    model = models.Model(run_args.num_hidden_units_1, run_args.num_hidden_units_2).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    stats = Stats([])
    return model, optimizer, stats


def save_checkpoint(path, model, optimizer, stats, run_args=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stats": stats,
            "run_args": run_args,
        },
        path,
    )
    logging.info(f"Saved checkpoint to {path}")


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    model, optimizer, stats = init(checkpoint["run_args"], device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    stats = checkpoint["stats"]
    run_args = checkpoint["run_args"]
    return model, optimizer, stats, run_args


Stats = collections.namedtuple("Stats", ["losses"])


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
