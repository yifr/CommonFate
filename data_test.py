from data_loader import SceneLoader
from torch.utils.data import DataLoader
import os
from models import cnn
import sys
import torchvision.models as models

cwd = os.getcwd()
dl = SceneLoader(root_dir=cwd + '/scenes', device='cpu')
s0 = dl.get_scene(0)

block = models.resnet.Bottleneck
model = cnn.ResNet(block, [3, 3, 3, 3], 4)

out = model(s0['frame'])

print(out.shape)
