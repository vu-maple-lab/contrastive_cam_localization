import torch 
from torch.utils.data import DataLoader 
import argparse 
from pathlib import Path 

from dataset import EndoscopicDataset
from model import Encoder
from losses import loss_fn

def train(args):
    data_dir = Path(args.data_dir)
    num_epochs = args.num_epochs 
    batch_size = args.batch_size
    num_negative_samples = args.num_neg
    latent_size = args.latent_size 

    if not data_dir.exists():
        raise Exception('data_directory does not exist! what the heck man, cmon') 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = EndoscopicDataset(data_dir=data_dir, num_negative_samples=num_negative_samples)
    model = Encoder(latent_size=latent_size).to(device)
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)

    # anchor_img, positive_img, positive_cam_diff, negative_imgs, negative_cam_diffs = next(iter(dataloader))
    breakpoint()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--num_neg', default=10, type=int)
    parser.add_argument('--latent_size', default=128, type=int)
    args = parser.parse_args()
    train(args)