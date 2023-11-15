import torch 
import torch.nn as nn 

def loss_fn(z_anchor, z_positive, z_negative, b=0.1): 
    cos_sim = nn.CosineSimilarity()
    return -cos_sim(z_anchor, z_positive) + b * torch.sum(cos_sim(z_anchor, z_negative), dim=2)