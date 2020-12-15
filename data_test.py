from data_loader import SceneLoader
from torch.utils.data import DataLoader
import os
import sys
cwd = os.getcwd()
dl = DataLoader(SceneLoader(root_dir=cwd + '/scenes', device='cpu'))
s0 = iter(dl).next()
print([(key, s0[key].shape) for key in s0.keys()])
