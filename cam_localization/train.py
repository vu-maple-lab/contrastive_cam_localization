import torch 
from torch.utils.data import DataLoader 
import argparse 
from pathlib import Path 
import os 
import matplotlib.pyplot as plt 
import numpy as np 
import math 
from tqdm import tqdm 
import torchvision.transforms as T
from torchvision.transforms import v2

from dataset import EndoscopicDataset
from model import Encoder
from losses import loss_fn

def train(args):
    data_dir = Path(args.data_dir)
    num_epochs = args.num_epochs 
    batch_size = args.batch_size
    lr = args.lr
    num_negative_samples = args.num_neg
    latent_size = args.latent_size 

    if not data_dir.exists():
        raise Exception('data_directory does not exist! what the heck man, cmon') 
    
    transforms = v2.Compose([
        T.ToPILImage(),
        v2.RandomVerticalFlip(p = 0.2),
        v2.RandomHorizontalFlip(p = 0.2),
        v2.RandomApply(transforms=[
        v2.RandomRotation(degrees=(0,180))],p=0.2),
        v2.RandomApply(transforms=[
        v2.GaussianBlur(kernel_size=(7,7))],p=0.2),
        v2.RandomApply(transforms=[
        v2.ElasticTransform(alpha=100.)],p=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = EndoscopicDataset(data_dir=data_dir, num_negative_samples=num_negative_samples, rendered_transforms=transforms, styletransfer_transforms=transforms)
    dataset_test = EndoscopicDataset(data_dir=data_dir, train=False, num_negative_samples=num_negative_samples, rendered_transforms=transforms, styletransfer_transforms=transforms)
    model = Encoder(latent_size=latent_size).to(device)
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)
    dataloader_test = DataLoader(dataset=dataset_test, shuffle=True, batch_size=batch_size)

    steps = math.floor(len(dataset) / batch_size)
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps)

    cos_loss = torch.nn.CosineEmbeddingLoss()
    # target = -torch.ones((batch_size, 1 + num_negative_samples))
    # target[:,0] = 1.0

    # anchor_img, positive_img, positive_cam_diff, negative_imgs, negative_cam_diffs = next(iter(dataloader))
    training_loss = []
    testing_loss = []
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        
        for i, (anchor_img, positive_img, positive_cam_diff, negative_imgs, negative_cam_diffs) in enumerate(tqdm(dataloader)):

            anchor_img = anchor_img.float().to(device)
            positive_img = positive_img.float().to(device)
            positive_cam_diff = positive_cam_diff.float().to(device)
            negative_cam_diffs = negative_cam_diffs.float().to(device)

            z_negatives = []
            z_anchor = model(anchor_img)
            z_positive = model(positive_img)
            for j in range(num_negative_samples):
                z_negatives.append(model(negative_imgs[j].float().to(device)))

            loss_val = loss_fn(cos_loss, batch_size, z_anchor, z_positive, z_negatives, device)

            optim.zero_grad()
            loss_val.backward()
            optim.step()
            scheduler.step()
            running_loss += loss_val.item()

        avg_loss_train = running_loss / (i+1)
        running_loss = 0.0
        model.eval()

        for i, (anchor_img, positive_img, positive_cam_diff, negative_imgs, negative_cam_diffs) in enumerate(tqdm(dataloader_test)):
            anchor_img = anchor_img.float().to(device)
            positive_img = positive_img.float().to(device)
            positive_cam_diff = positive_cam_diff.float().to(device)
            negative_imgs = negative_imgs.float().to(device)
            negative_cam_diffs = negative_cam_diffs.float().to(device)

            z_negatives = []
            z_anchor = model(anchor_img)
            z_positive = model(positive_img)
            for j in range(num_negative_samples):
                z_negatives.append(model(negative_imgs[j]))
            loss_val = loss_fn(cos_loss, batch_size, z_anchor, z_positive, z_negatives, device)
            running_loss += loss_val.item()
        
        avg_loss_test = running_loss / (i+1)
        
        training_loss.append(avg_loss_train)
        testing_loss.append(avg_loss_test)
        print(f'FOR EPOCH {epoch+1}/{num_epochs}, average training loss: {avg_loss_train}, average testing loss: {avg_loss_test}')

    # save visualizations and outputs
    if not Path('./figs').exists():
        os.mkdir('./figs')
    if not Path('./logs').exists():
        os.mkdir('./logs')

    np.save('./logs/training_loss.npy', np.array(training_loss))
    np.save('./logs/testing_loss.npy', np.array(testing_loss))
    torch.save({
        'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': optim.state_dict(), 
        'scheduler_state_dict': scheduler.state_dict()
    }, './logs/checkpoint.pt')

    plt.figure()
    plt.plot(np.array(training_loss))
    plt.title('Training loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.savefig(str('figs' / 'training_loss.png'))

    # training accuracy 
    plt.figure()
    plt.plot(np.array(testing_loss))
    plt.title('Testing Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Testing Loss')
    plt.savefig(str('figs' / 'testing_loss.png'))

    print('yay')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--lr', default=1., type=float)
    parser.add_argument('--num_neg', default=10, type=int)
    parser.add_argument('--latent_size', default=128, type=int)
    args = parser.parse_args()
    train(args)