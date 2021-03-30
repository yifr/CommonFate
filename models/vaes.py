import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
        def __init__(self, img_channels=1, n_latents=5, h_dim=256):
                super(VAE, self).__init__()

                self.encoder = nn.Sequential(
                                nn.Conv2d(img_channels, 32, kernel_size=4, stride=2),
                                nn.ReLU(),
                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                nn.ReLU(),
                                nn.Conv2d(64, 128, kernel_size=4, stride=2),
                                nn.ReLU(),
                                nn.Conv2d(128, 256, kernel_size=4, stride=2),
                                )
