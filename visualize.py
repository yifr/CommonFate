import os
import torch
from tqdm import tqdm
import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.image as mpimg

# Add paths so Blender can do relative imports
import sys
import pathlib

cwd = os.getcwd()
generator_path = os.path.join(cwd, "generator")
sys.path.append(cwd)
sys.path.append(generator_path)

from data_loader import SceneLoader
from models import cnn

from generator import utils
from generator import superquadrics
from generator import BlenderArgparse
from generator.render_scenes import BlenderScene
from generator.textures import dot_texture

import argparse

parser = BlenderArgparse.ArgParser()

parser.add_argument("--model_path", required=True, type=str, help="Path to saved model")
parser.add_argument(
    "--save_dir",
    required=True,
    type=str,
    default="media/",
    help="where to save results",
)
parser.add_argument(
    "--root_scene_dir",
    required=True,
    type=str,
    nargs="+",
    help="Directory for data - listing more than one will create a joint data loader",
)

parser.add_argument(
    "--conv_dims", type=int, default=2, help="Conv dims -- if 3d conv set this to 3"
)
parser.add_argument(
    "--n_scenes", type=int, default=1000, help="number of scenes for data loader"
)
parser.add_argument("--n_frames", type=int, default=20, help="Frames per scene")
parser.add_argument(
    "--viz_types",
    type=str,
    nargs="+",
    default=["predictions", "ground_truth"],
    help="What to visualize",
)
parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
parser.add_argument("--random_seed", type=int, default=42)

ROOT_SCENE_DIR = []


def render_ground_truth(scene_dir):
    # Initialize scene
    scene = BlenderScene(scene_dir, n_frames=20)
    scene.set_light_source("SUN", [7.35, -6.9, 0], [1.5707, 0, 0.7853])
    obj = scene.load_mesh()

    # Retrieve ground truth data
    gt_data = np.load(os.path.join(scene_dir, "data.npy"), allow_pickle=True).item()

    # Animate and render scene
    scene.generate_rotation(obj)
    scene.render(output_dir="ground_truth")


def render_from_predictions(
    scene_loader, models, scene_dir, textured=False, device="cpu"
):
    """
    Given a model and a scene, render predictions
    Params:
    -------
        models: :dict: dict containing {prediction_type: model}
        root_scene_dir: :str: root directory for scene data
        scene_dir: :int: scene directory
        textured: :bool: whether or not to render texture or just underlying mesh
    """
    scene = BlenderScene(scene_dir, n_frames=20)
    print(f"Initialized scene for {scene_dir}. Exporting predictions...")

    predictions = load_predictions(scene_loader, models, scene_dir, device=device)
    pred_types = models.keys()

    # If we're visualizing shape predictions, create new mesh
    obj = None
    print("Loading shape...")
    if "shape_params" in pred_types:
        shape_pred = predictions["shape_params"][:, 0]

        shape_params = {"mesh_0": {"exponents": shape_pred, "scaling": [1, 1, 1, 2]}}
        print("Predicted shape: ", shape_pred)
        obj = scene.create_mesh(shape_params=shape_params)
    else:
        print("Loading mesh")
        obj = scene.load_mesh()

    print(f"Loaded mesh: {obj}")

    # Animate predicted rotations
    if "rotation" in pred_types:
        # Convert predicted 6d rotation to quaternion
        quaternions = ortho6d_to_quaternion(predictions["rotation"])
    else:
        data = np.load(os.path.join(scene_dir, "data.npy"), allow_pickle=True).item()
        quaternions = data[0]["quaternion"]

    scene.generate_rotation(obj)
    prediction_dir = os.path.join(scene_dir, "predictions")
    if not os.path.exists(prediction_dir):
        os.mkdir(prediction_dir)

    print("Rendering predictions")
    scene.render(output_dir="predictions")


def load_predictions(scene_loader, models, scene_dir, device="cpu", overwrite=True):
    """
    Loads predictions if they exist, creates predictions file if they don't exist
    Params
    ------
        models: :dict: dict containing {prediction_type: model_path}
    Returns:
        predictions from given model as numpy vectors
    """
    pred_path = os.path.join(scene_dir, "predictions.npy")

    predictions = {}

    if os.path.exists(pred_path) and not overwrite:
        predictions = np.load(pred_path, allow_pickle=True).item()

    for pred_type in models.keys():
        model = models[pred_type]

        # output predictions if they don't exist
        if pred_type not in predictions.keys():
            preds = create_predictions(
                scene_loader,
                model,
                scene_dir=scene_dir,
                pred_type=pred_type,
                device=device,
                save=True,
            )
            predictions[pred_type] = preds

    return predictions


def create_predictions(
    scene_loader, model, scene_dir, device="cpu", pred_type="rotation", save=True
):
    """
    Loads model and gets predicted values for a given scene.
    Params
    ------
        model_path: :str: path to model to load
        root_scene_dir: :str: root directory for scene data
        pred_type: :str: indicates the data to be predicted
                         if save == True, pred_type will be used as the key when saving predictions
        save: :bool: whether or not to save predictions to scene directory

    Returns:
        predictions from given model as numpy vectors
    """

    # Load model and data to predict
    transforms = model.get_transforms()
    as_rgb = False if str(model) != "ResNet" else True

    print("Loading data...")
    scene_data = scene_loader.get_scene(scene_dir)

    if pred_type not in scene_data.keys():
        raise ValueError(
            f"Cannot find key: {pred_type} in scene data keys: {scene_data.keys()}"
        )

    input_data = scene_data["frame"]
    if args.conv_dims == 3:
        input_data = input_data.reshape(1, 20, 256, 256).unsqueeze(0)

    # Push data through model
    model.eval()
    print("Data loaded. Generating model predictions...")

    predictions = model(input_data)

    predictions = predictions.detach().cpu().numpy()
    if save:
        save_file = os.path.join(scene_dir, "predictions.npy")
        if os.path.exists(save_file):
            save_data = np.load(save_file, allow_pickle=True).item()
            if not save_data.get(0):
                save_data[0] = {}
            save_data[0][pred_type] = predictions
        else:
            save_data = {0: {}}
            save_data[0] = {pred_type: predictions}

        np.save(save_file, save_data)

    return predictions


def stitch_prediction_video(scene_dir, save_dir, save_file, n_frames=20):
    os.makedirs(save_dir, exist_ok=True)
    outfile = os.path.join(save_dir, save_file)

    fig, (gt_fig, stimuli_fig, predicted_fig) = plt.subplots(1, 3, figsize=(16, 12))

    gt_fig.set_title("Ground Truth")
    stimuli_fig.set_title("Stimuli")
    predicted_fig.set_title("Predicted Shape")
    gt_fig.axis("off")
    stimuli_fig.axis("off")
    predicted_fig.axis("off")

    gt_path = os.path.join(scene_dir, "ground_truth")
    stim_path = os.path.join(scene_dir, "images")
    pred_path = os.path.join(scene_dir, "predictions")

    frames = []
    print("Compiling frames for animation...")
    for frame_idx in tqdm(range(1, n_frames)):
        gt_image = mpimg.imread(os.path.join(gt_path, f"img_{frame_idx:04d}.png"))
        stim_image = mpimg.imread(os.path.join(stim_path, f"img_{frame_idx:04d}.png"))
        pred_image = mpimg.imread(os.path.join(pred_path, f"img_{frame_idx:04d}.png"))

        cmap = "gray"
        gt_frame = gt_fig.imshow(gt_image, cmap=cmap, animated=True)
        stim_frame = stimuli_fig.imshow(stim_image, cmap=cmap, animated=True)
        pred_frame = predicted_fig.imshow(pred_image, cmap=cmap, animated=True)

        if frame_idx == 1:
            gt_fig.imshow(gt_image, cmap=cmap)
            stimuli_fig.imshow(stim_image, cmap=cmap)
            predicted_fig.imshow(pred_image, cmap=cmap)

        frames.append((gt_frame, stim_frame, pred_frame))

    print("Creating gif")
    ani = animation.ArtistAnimation(fig, frames)
    # FFMpegWriter = animation.writers['ffmpeg']
    # writer = FFMpegWriter(fps=25)
    writer = animation.PillowWriter(fps=10)
    ani.save(outfile, writer=writer)
    print("Done.")


def main():
    ROOT_SCENE_DIR = args.root_scene_dir

    device = args.device
    viz_types = args.viz_types
    seed = args.random_seed

    model = cnn.ShapeNet(out_size=4, conv_dims=args.conv_dims).to(device)
    model_path = os.path.join(os.getcwd(), args.model_path)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    scene_loader = SceneLoader(
        root_dirs=args.root_scene_dir,
        n_scenes=args.n_scenes,
        n_frames=args.n_frames,
        device=device,
        transforms=model.get_transforms(),
        seed=seed,
    )

    test_idxs = scene_loader.test_idxs
    scenes = np.concatenate((test_idxs[:5], test_idxs[-5:]), axis=0)
    print(scenes)
    for scene_num in scenes:
        print("Rendering scene: ", scene_num)
        if "predictions" in viz_types:
            print("Initializing model...")
            scene_dir = scene_loader.get_scene_dir(scene_num)
            render_from_predictions(
                scene_loader,
                models={"shape_params": model},
                scene_dir=scene_dir,
                device=device,
            )

        if "ground_truth" in viz_types:
            print("Rendering Ground truth frames")
            render_ground_truth(scene_dir=scene_dir)

        stitch_prediction_video(
            scene_dir,
            save_dir=args.save_dir,
            save_file=f"scene_{scene_num}_predictions.gif",
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main()
