import os
import sys
import json
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class Scene(Dataset):
    def __init__(self, scene_dir, device='cuda', transforms=None, n_frames=100, img_size=128, as_rgb=False):
        """
        Encapsulates a single scene
        """
        self.scene_dir = scene_dir
        self.image_dir = os.path.join(self.scene_dir, 'images')
        self.n_frames = n_frames
        self.img_size = img_size

        if transforms == None:
            transforms = T.Compose([T.Resize(self.img_size), T.ToTensor()])
        self.transforms = transforms

        self.device = device
        self.as_rgb = as_rgb # Whether to load images as 3 channel rgb images

        # Load rotation parameters
        scene_data = os.path.join(self.scene_dir, 'data.npy')
        data_dict = np.load(scene_data, allow_pickle=True).item()[0]

        self.shape_params = self.get_exponents()
        self.rotations = data_dict['quaternion']
        self.angle = data_dict['angle']
        self.axis = data_dict['axis']
        self.params = self.get_exponents()

    def __len__(self):
        """
        Return number of frames in video
        """
        return self.n_frames

    def __getitem__(self, idx):
        """
        Return frame image, shape and rotation parameters for
        a given frame
        """
        frame_idx = idx + 1 # frame images start from 0001

        filename = os.path.join(self.image_dir, 'img_%04d.png' % frame_idx)
        img_raw = Image.open(filename)

        # If we're using a pre-trained model, create 3 channels (images are stacked 1 channel Black and White)
        if self.as_rgb:
            img_raw = np.array(img_raw)
            img_raw = np.repeat(img_raw[..., np.newaxis], 3, -1)

        img = self.transforms(img_raw).to(self.device)

        rotation = torch.tensor(self.rotations[idx], dtype=torch.float).to(self.device)
        angle = torch.tensor(self.angle[idx]).to(self.device)
        axis = torch.tensor(self.axis[idx]).to(self.device)

        return {'frame': img, 'rotation': rotation, 
                'angle': angle, 'axis': axis, 'shape_params': self.params}

    def get_exponents(self):
        """
        Extracts exponents and scaling from params.json file in scene dir and returns as torch tensor
        Args:
            scene_dir (string): directory containing params.txt file
        """
        param_file = os.path.join(self.scene_dir, 'params.json')
        with open(param_file, 'r') as f:
            param_data = json.load(f)
        exponents = torch.tensor(param_data['mesh_0']['exponents'], dtype=torch.float32, device=self.device)
        return exponents

class SceneLoader():
    """
    Describes a dataset over scenes. Each scene directory contains
    an image folder with N frames of a superquadric in rotation.
    There's also a numpy dict with entries for the quaternion
    rotation, angle, (position and translation - right now these are static)
    of the shape. Lastly, we have a small text file with the exponents
    to recreate a superquadric, given the formula:
        $$ |x/A|^e_1 + |y/B|^e_2 + |z/C|^e_3 = 1 $$
    assuming A = B = C = 1
    """

    def __init__(self, root_dirs, transforms=None, device='cuda',
                 n_scenes=0, n_frames=20, img_size=256,
                 batch_size=100, train_size=0.8, as_rgb=False, seed=42):
        """
        Args:
            root_dir (string): root directory for the generated scenes
            img_transform (callable, optional): Optional transform to be applied to images
            n_scenes (int, optional): Total number of scenes in dataset (will be counted automatically if 0)
        """
        for scene_dir in root_dirs:
            if not os.path.exists(scene_dir):
                 raise ValueError(f'Data directory: {scene_dir} does not exist!')

        self.root_dirs = root_dirs
        self.transforms = transforms
        self.device = device
        self.batch_size = batch_size # Batch size is defined over frames per scene
        self.seed = seed
        
        if n_scenes == 0:
            self.n_scenes = len(os.listdir(root_dirs[0]))
        else:
            self.n_scenes = n_scenes

        self.n_frames = n_frames
        self.img_size = img_size
        self._train = True

        self.as_rgb = as_rgb
        self.train_size = train_size
        self.train_test_split(train_size)
        print(f'''Data Loader initialized:
                    \troot_dirs: {self.root_dirs},
                    \tscenes: {self.n_scenes},
                    \tframes: {self.n_frames},
                    \ttrain/test: {len(self.train_idxs), len(self.test_idxs)}''')

    def train_test_split(self, train_size=0.8, test_size=0.2):
        n_train = int(train_size * self.n_scenes)
        np.random.seed(self.seed)
        self.train_idxs = np.random.choice(self.n_scenes, n_train, replace=False)
        self.test_idxs = np.delete(np.arange(0, self.n_scenes, 1), self.train_idxs) # remaining indexes used for test

    def __len__(self):
        """
        Returns total number of scenes in root directory
        """
        return self.n_scenes    

    def get_scene_dir(self, idx):
        scenes_per_dir = int(self.n_scenes / len(self.root_dirs))
        root_dir_idx = int(idx / scenes_per_dir)
        scene_dir = self.root_dirs[root_dir_idx]
        scene_idx = idx % scenes_per_dir
        return os.path.join(scene_dir, f'scene_{scene_idx:03d}')

    def get_scene(self, scene_dir):
        """
        scene_dir: either int specifying scene_idx or full scene_dir path
        Compiles scene data for a scene at a given index
        """
        if type(scene_dir) != str:
            scene_dir = self.get_scene_dir(scene_dir)
        
        scene = Scene(scene_dir, device=self.device,
                      n_frames=self.n_frames, img_size=self.img_size,
                      transforms=self.transforms, as_rgb=self.as_rgb)

        scene_loader = DataLoader(scene, batch_size=self.batch_size)
        data = iter(scene_loader).next()
        return data

