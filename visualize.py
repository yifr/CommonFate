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
generator_path = os.path.join(cwd, 'generator')
sys.path.append(cwd)
sys.path.append(generator_path)

from data_loader import SceneLoader
from models import cnn

from generator import utils
from generator import superquadrics
from generator import BlenderArgparse
from generator.render_scenes import BlenderScene
from generator.textures import dot_texture

ROOT_SCENE_DIR = 'scenes/'

def ortho6d_to_quaternion(ortho6d):
    loss = cnn.Loss()
    rotation_matrices = loss.rotation_matrix_from_6d(ortho6d)
    rotation_matrices = rotation_matrices.detach().numpy()
    print('Predicted Rotation Matrices: ', rotation_matrices)
    quaternions = np.zeros(shape=(ortho6d.shape[0], 4))
    for i, mat in enumerate(rotation_matrices):

        quat = Quaternion(matrix=mat, atol=0.0001)
        quaternions[i] = quat.elements

    return quaternions

def render_ground_truth(scene_num):
    # Initialize scene
    scene_dir = os.path.join(ROOT_SCENE_DIR, f'scene_{scene_num:03d}')
    scene = BlenderScene(scene_dir)
    obj = scene.load_mesh()

    # Retrieve ground truth data
    gt_data = np.load(os.path.join(scene_dir, 'data.npy'), allow_pickle=True).item()
    rotation_key = 'quaternion' if 'quaternion' in gt_data.keys() else 'rotation'
    rotation = gt_data[rotation_key]

    # Animate and render scene
    scene.rotate(obj, rotation)
    scene.render(output_dir='ground_truth')

def render_from_predictions(models={'rotation': None}, scene_num=0, textured=False):
    """
    Given a model and a scene, render predictions
    Params:
    -------
        models: :dict: dict containing {prediction_type: model}
        root_scene_dir: :str: root directory for scene data
        scene_num: :int: scene number
        textured: :bool: whether or not to render texture or just underlying mesh
    """
    scene_dir = os.path.join(ROOT_SCENE_DIR, f'scene_{scene_num:03d}')
    scene = BlenderScene(scene_dir)
    print(f'Initialized scene for {scene_dir}. Exporting predictions...')

    predictions = load_predictions(models, scene_num)
    pred_types = models.keys()

    # If we're visualizing shape predictions, create new mesh
    obj = None
    if 'shape_params' in pred_types:
        shape_pred = list(np.mean(predictions['shape_params'].detach().numpy(), axis=0))
        shape_pred.append(shape_pred[0])
        print('Predicted shape: ', shape_pred)
        obj = scene.create_mesh(shape_params=shape_pred)
        print('Creating new object')
    else:
        print('Loading mesh')
        obj = scene.load_mesh()

    print(f'Loaded mesh: {obj}')

    # Animate predicted rotations
    if 'rotation' in pred_types:
        # Convert predicted 6d rotation to quaternion
        quaternions = ortho6d_to_quaternion(predictions['rotation'])
    else:
        data = np.load(os.path.join(scene_dir, 'data.npy'), allow_pickle=True).item()
        quaternions = data['rotation']

    scene.rotate(obj, quaternions)
    prediction_dir = os.path.join(scene_dir, 'predictions')
    if not os.path.exists(prediction_dir):
        os.mkdir(prediction_dir)

    print('Rendering predictions')
    scene.render(output_dir='predictions')

def load_predictions(models={'rotation': None}, scene_num=0, overwrite=True):
    """
    Loads predictions if they exist, creates predictions file if they don't exist
    Params
    ------
        models: :dict: dict containing {prediction_type: model_path}
        scene_num: :int: scene number (will be used to load data from given scene)
    Returns:
        predictions from given model as numpy vectors
    """
    scene_dir = os.path.join(ROOT_SCENE_DIR, f'scene_{scene_num:03d}')
    pred_path = os.path.join(scene_dir, 'predictions.npy')

    predictions = {}

    if os.path.exists(pred_path) and not overwrite:
        predictions = np.load(pred_path, allow_pickle=True).item()

    for pred_type in models.keys():
        model = models[pred_type]

        # output predictions if they don't exist
        if pred_type not in predictions.keys():
            preds = create_predictions(model, scene_num=scene_num, pred_type=pred_type, save=True)
            predictions[pred_type] = preds

    return predictions

def create_predictions(model=None, device='cpu', scene_num=0, pred_type='rotation', save=True):
    """
    Loads model and gets predicted values for a given scene.
    Params
    ------
        model_path: :str: path to model to load
        scene_num: :int: scene number (will be used to load data from given scene)
        root_scene_dir: :str: root directory for scene data
        pred_type: :str: indicates the data to be predicted
                         if save == True, pred_type will be used as the key when saving predictions
        save: :bool: whether or not to save predictions to scene directory

    Returns:
        predictions from given model as numpy vectors
    """

    # Load model and data to predict
    transforms = model.get_transforms()
    as_rgb = False
    if str(model) == 'ResNet':
        as_rgb = True

    print('Loading data...')
    # print(f'root dir: {ROOT_SCENE_DIR}, pred_type: {pred_type}, scene_num: {scene_num}')
    scene_loader = SceneLoader(ROOT_SCENE_DIR, device=device, as_rgb=as_rgb, transforms=transforms)
    scene_data = scene_loader.get_scene(scene_num)

    if pred_type not in scene_data.keys():
        raise ValueError(f'Cannot find key: {pred_type} in scene data keys: {scene_data.keys()}')

    input_data = scene_data['frame']

    # Push data through model
    model.eval()
    print('Data loaded. Generating model predictions...')
    predictions = model(input_data)
    predicted = predictions.detach().cpu().numpy()

    if save:
        save_file = os.path.join(ROOT_SCENE_DIR, 'scene_%03d' % scene_num, 'predictions.npy')
        if os.path.exists(save_file):
            save_data = np.load(save_file, allow_pickle=True).item()
            save_data[pred_type] = predictions
        else:
            save_data = {pred_type: predictions}

        np.save(save_file, save_data)

    return predictions


def stitch_prediction_video(scene_num, n_frames=100):
    scene_dir = os.path.join(ROOT_SCENE_DIR, f'scene_{scene_num:03d}')

    fig, (gt_fig, stimuli_fig, predicted_fig) = plt.subplots(1, 3, figsize=(20, 20))

    gt_fig.set_title('Ground Truth (shape/rotation)')
    stimuli_fig.set_title('Stimuli')
    predicted_fig.set_title('Predicted (shape/rotation)')
    gt_fig.axis('off')
    stimuli_fig.axis('off')
    predicted_fig.axis('off')

    gt_path = os.path.join(scene_dir, 'ground_truth')
    stim_path = os.path.join(scene_dir, 'images')
    pred_path = os.path.join(scene_dir, 'predictions')

    frames = []
    print('Compiling frames for animation...')
    for frame_idx in tqdm(range(1, n_frames)):
        gt_image = mpimg.imread(os.path.join(gt_path, f'img_{frame_idx:04d}.png'))
        stim_image = mpimg.imread(os.path.join(stim_path, f'img_{frame_idx:04d}.png'))
        pred_image = mpimg.imread(os.path.join(pred_path, f'img_{frame_idx:04d}.png'))

        cmap = 'gray'
        gt_frame = gt_fig.imshow(gt_image, cmap=cmap, animated=True)
        stim_frame = stimuli_fig.imshow(stim_image, cmap=cmap, animated=True)
        pred_frame = predicted_fig.imshow(pred_image, cmap=cmap, animated=True)

        if frame_idx == 1:
            gt_fig.imshow(gt_image, cmap=cmap)
            stimuli_fig.imshow(stim_image, cmap=cmap)
            predicted_fig.imshow(pred_image, cmap=cmap)

        frames.append((gt_frame, stim_frame, pred_frame))

    print('Creating gif')
    ani = animation.ArtistAnimation(fig, frames)
    # FFMpegWriter = animation.writers['ffmpeg']
    # writer = FFMpegWriter(fps=25)
    writer = animation.PillowWriter(fps=25)
    ani.save('predictions.gif', writer=writer)
    print('Done.')

def main():
    viz_types = [] #['predictions'] #['ground_truth']
    scene_num = 25

    if 'predictions' in viz_types:
        print('Initializing model...')
        rotation_model = cnn.SimpleCNN(out_size=2)

        model_path = os.path.join(os.getcwd(), 'saved_models/shapenet.pt')
        model = torch.load(model_path, map_location=torch.device('cpu'))


        #rotation_model = cnn.SimpleCNN()
        render_from_predictions(models={'shape_params': rotation_model}, scene_num=scene_num)

    if 'ground_truth' in viz_types:
        print('Rendering Ground truth frames')
        render_ground_truth(scene_num=scene_num)

    stitch_prediction_video(scene_num)

if __name__=='__main__':
    main()
