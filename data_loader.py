import os
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Scene(Dataset):
    def __init__(self, root_dir, scene_number, device='cuda', transforms=None, n_frames=100, img_size=128):
        """
        Encapsulates
        """
        self.root_dir = root_dir
        self.scene_dir = os.path.join(root_dir, 'scene_%03d' % scene_number)
        self.image_dir = os.path.join(self.scene_dir, 'images')
        self.shape_params = self.get_exponents()
        self.n_frames = n_frames
        self.img_size = img_size
        self.transforms = transforms
        self.device = device

        # Load rotation parameters
        scene_data = os.path.join(self.scene_dir, 'data.npy')
        data_dict = np.load(scene_data, allow_pickle=True).item()
        self.rotations = data_dict['rotation']
        self.translation = data_dict['translation']
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
        if self.img_size == 128:
            filename = 'img_128_%04d.png' % frame_idx
        elif self.img_size == 64:
            filename = 'img_128_%04d_sm.png' % frame_idx
        else:
            filename = 'img_%04d.png' % frame_idx

        filename = os.path.join(self.image_dir, filename)
        img_raw = Image.open(filename)
        img = transforms.ToTensor()(img_raw).to(self.device)
        rotation = torch.tensor(self.rotations[idx]).to(self.device)
        translation = torch.tensor(self.translation[idx]).to(self.device)
        angle = torch.tensor(self.angle[idx]).to(self.device)
        axis = torch.tensor(self.axis[idx]).to(self.device)

        return {'frame': img, 'rotation': rotation, 'translation': translation,
                'angle': angle, 'axis': axis, 'shape_params': self.params}

    def get_exponents(self):
        """
        Extracts exponents from params.txt file in scene dir (exponents are stored in a
        .txt file because of something called "Tech Debt", where stupid choices are made
        early on and then you just stick to them.)
        Args:
            scene_dir (string): directory containing params.txt file
        """
        param_file = os.path.join(self.scene_dir, 'params.txt')
        param_data = open(param_file, 'r').read()
        params = torch.tensor([float(x.split(': ')[1]) for x in param_data.split('\n')], dtype=torch.float32)
        return params

class SceneLoader():
    """
    Describes a dataset over scenes. Each scene directory contains
    an image folder with N frames of a superquadric in rotation.
    There's also a numpy dict with entries for the quaternion
    rotation, angle, (position and translation - right now these are static)
    of the shape. Lastly, we have a small text file with the exponents
    to recreate a superquadric, given the basic formula:
        $$ |x/A|^e_1 + |y/B|^e_2 + |z/C|^e_3 = 1
    assuming A = B = C = 1
    """

    def __init__(self, root_dir, transforms=None, device='cuda',
                 n_scenes=0, n_frames=100, img_size=128,
                 batch_size=100, train_size=0.8):
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
        self.transforms = transforms
        self.device = device
        self.batch_size = batch_size # Batch size is defined over frames per scene

        if n_scenes == 0:
            self.n_scenes = len(os.listdir(root_dir))
        else:
            self.n_scenes = n_scenes

        self.n_frames = n_frames
        self.img_size = img_size
        self._train = True

        self.train_size = train_size
        self.train_test_split(train_size)

    def train_test_split(self, train_size=0.8):
        n_train = int(train_size * self.n_scenes)
        self.train_idxs = np.random.choice(self.n_scenes, n_train, replace=False)
        self.test_idxs = np.delete(np.arange(0, self.n_scenes, 1), self.train_idxs)

    def eval(self):
        self._train = False

    def train(self):
        self._train = True

    def __len__(self):
        """
        Returns total number of scenes in root directory
        """
        return self.n_scenes


    def next(self, idx):
        """
        Compiles scene data for a scene at a given index
        """
        scene = Scene(self.root_dir, idx, device=self.device,
                      n_frames=self.n_frames, img_size=self.img_size,
                      transforms=self.transforms)


        scene_loader = DataLoader(scene, batch_size=self.batch_size)
        data = iter(scene_loader).next()
        return data
