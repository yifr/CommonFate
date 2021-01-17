from data_loader import SceneLoader
from torch.utils.data import DataLoader
import os
from models import cnn
import sys
import torchvision.models as models

cwd = os.getcwd()
dl = SceneLoader(root_dir=cwd + '/scenes', device='cpu', transforms=None)
s0 = dl.get_scene(0)

for i in range(len(dl.train_idxs)):
    data = dl.get_scene(i)
    print(data['rotation'].shape)
