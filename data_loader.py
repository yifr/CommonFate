import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class SceneLoader(Dataset):
    """
    Describes a dataset over scenes. Each scene directory contains
    an image folder with N frames of a superquadric in rotation

    There's also a numpy dict with entries for the quaternion
    rotation, angle, (position and translation - right now these are static)
    of the shape. Lastly, we have a small text file with the exponents
    to recreate a superquadric, given the basic formula:
        $$ |x/A|^e_1 + |y/B|^e_2 + |z/C|^e_3 = 1
    assuming A = B = C = 1
    """

    def __init__(self, root_dir, img_transform=None, n_scenes=0):
        """
        Args:
            root_dir (string): root directory for the generated scenes
            img_transform (callable, optional): Optional transform to be applied to images
            n_scenes (int, optional): Total number of scenes in dataset (will be counted automatically if 0)
        """
        if not os.path.exists(root_dir):
            print('Data directory {} does not exist!'.format(root_dir))
            raise IOError

        self.root_dir = root_dir
        self.img_transform = img_transform
        self.n_scenes = n_scenes

        if n_scenes == 0:
            self.n_scenes = len(os.listdir(root_dir))


    def __len__(self):
        """
        Returns total number of scenes in root directory
        """
        return self.n_scenes


    def get_exponents(self, scene_dir):
        """
        Extracts exponents from params.txt file in scene dir (exponents are stored in a
        .txt file because of something called "Tech Debt", where stupid choices are made
        early on and then you just stick to them.)
        Args:
            scene_dir (string): directory containing params.txt file
        """
        param_file = os.path.join(scene_dir, 'params.txt')
        param_data = open(param_file, 'r').read()
        params = torch.tensor([float(x.split(': ')[1]) for x in d.split('\n'), dtype='float32'])
        return params

    def __getitem__(self, idx):
        """
        Compiles scene data for a scene at a given index
        """
        scene_dir = os.path.join(self.root_dir + 'scene_%03d'%idx)

        # Get shape parameters
        params = self.get_exponents(scene_dir)

        # Load Images
        im
