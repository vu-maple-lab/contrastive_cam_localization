import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

class Encoder(nn.Module):

    def __init__(self, latent_size):
        # load pretrained from imagenet :D
        # self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.layer1 = nn.Linear(in_features=1000, out_features=512)
        self.layer2 = nn.Linear(in_features=512, out_features=latent_size)
    
    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x