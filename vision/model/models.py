

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, ResNet50_Weights

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)
    def forward(self, x):
        return torch.reshape(x, (-1, *self.shape))

class DetectionNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        inner_channels = 1024
        self.depth = 2 *(4 +1) + 20
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
            nn.Linear(7 * 7 * inner_channels, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, 7 * 7 * self.depth),
        )
    def forward(self, x):
        x = self.model(x)
        return torch.reshape(x, (-1, 7, 7, self.depth))
 
class ImageResNet(nn.Module):
    def __init__(self):
        super(ImageResNet, self).__init__()
        
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.requires_grad_(False)
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.model = nn.Sequential(
            backbone,
            Reshape(2048, 14, 14),
            DetectionNet(2048),
        )
    def forward(self, x):
        return self.model(x)
    
class ImageMaskNet(nn.Module):
    def __init__(self):
        super(ImageMaskNet, self).__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.requires_grad_(False)
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.features_backbone = backbone
        self.model = nn.Sequential(
            Reshape(2048, 14, 14),
            DetectionNet(2048),
        )
        self.mask_pred = nn.Sequential(
            Reshape(2, 448, 448),
            nn.ConvTranspose2d(2, 1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1, 21, kernel_size=1),
        )
    def forward(self, x):
        feature = self.features_backbone(x)
        boxs = self.model(feature)
        mask = self.mask_pred(feature)
        return boxs, mask
    