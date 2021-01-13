import os
import bpy
import torch
from numpy import np
from pyquaternion import Quaternion
from data_loader import SceneLoader

# Add the generator directory to the path for relative Blender imports
import sys
import pathlib

generator_path = os.path.join(str(pathlib.Path(__file__).parent.absolute()), 'generator')
sys.path.append(generator_path)

from generator import utils
from generator import superquadrics
from generator import BlenderArgparse
from generator import render_scenes
from generator.textures import dot_texture

ROOT_SCENE_DIR = 'scenes/'

def render_from_predictions(models={'rotation': 'rotation_model.pt'}, scene_num=0, textured=False):
    """
    Given a model and a scene, render predictions
    Params:
    -------
        models: :dict: dict containing {prediction_type: model_path}
        root_scene_dir: :str: root directory for scene data
        scene_num: :int: scene number 
        textured: :bool: whether or not to render texture or just underlying mesh
    """
    scene_dir = os.path.join(ROOT_SCENE_DIR, 'scene_%03d' % scene_num)
    predictions = load_predictions(models, scene_num)
    pred_types = models.keys()

    utils.delete_all(obj_type='MESH')

    # If we're visualizing shape predictions, load new mesh 
    if 'shape' in pred_types:
        render_scenes.add_superquadric_mesh(predictions['shape'])
    else:
        obj_file = os.path.join(scene_dir, 'mesh.obj')
        utils.load_obj(obj_file)

def render_from_shape(scene, shape, textured=False):
    pass


def load_predictions(models={'rotation': 'rotation_model.pt'}, scene_num=0):
    """
    Loads predictions if they exist, creates predictions file if they don't exist
    Params
    ------
        models: :dict: dict containing {prediction_type: model_path}
        scene_num: :int: scene number (will be used to load data from given scene)
    Returns: 
        predictions from given model as numpy vectors
    """

    pred_path = os.path.join(ROOT_SCENE_DIR, 'scene_%03d' % scene_num, 'predictions.npy')
    
    if not os.path.exists(pred_path):
        predictions = {}
    else:
        predictions = np.load(pred_path, allow_pickle=True).item()

    for pred_type in models.keys(): 
        model_path = models[pred_type]

        if not os.path.exists(model_path):
            raise ValueError(f'No model found at given path: {model_path}')

        # output predictions if they don't exist
        if pred_type not in predictions.keys():
            preds = create_predictions(model_path, scene_num, ROOT_SCENE_DIR, pred_type, save=True)
            predictions[pred_type] = preds

    return predictions

def create_predictions(model_path='model.pt', scene_num=0, root_scene_dir='scenes/', pred_type='rotation', save=True):
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
    if not os.path.exists(model_path):
        print('Model cannot be found at given path: ', model_path)
        sys.exit(1)

    # Load model and data to predict
    model = torch.load(model_path)
    scene_loader = SceneLoader(root_scene_dir)
    scene_data = SceneLoader.get_scene(scene_num)

    if pred_type not in scene_data.keys():
        print('Cannot find key: {} in scene data keys: {}'.format(pred_type, scene_data.keys()))
        sys.exit(1)

    pred_data = scene_data[pred_type]

    # Push data through model
    model.eval()
    predictions = model(pred_data)

    # Convert predictions into numpy vectors on cpu
    predicted = predictions.detach().cpu().numpy()

    if save:
        save_file = os.path.join(root_scene_dir, 'scene_%03d' % scene_num, 'predictions.npy')
        if os.path.exists(save_file):
            save_data = np.load(save_file, allow_pickle=True).item()
            save_data[pred_type] = predictions
        else:
            save_data = {pred_type: predictions}

        np.save(save_file, save_data)

    return predictions