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
    negative_loss_weight = args.negative_loss
    positive_loss_weight = args.positive_loss
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
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    dataloader_test = DataLoader(dataset=dataset_test, shuffle=True, batch_size=batch_size, drop_last=True)

    steps = math.floor(len(dataset) / batch_size)
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps)

    cos_loss = torch.nn.CosineEmbeddingLoss(reduce=False)

    training_loss = []
    testing_loss = []
        # save visualizations and outputs
    figs_path = './figs_' + str(positive_loss_weight) + '_' + str(negative_loss_weight)
    logs_path = './logs_' + str(positive_loss_weight) + '_' + str(negative_loss_weight)
    if not Path(figs_path).exists():
        os.mkdir(figs_path)
    if not Path(logs_path).exists():
        os.mkdir(logs_path)

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

            loss_val = loss_fn(cos_loss, batch_size, z_anchor, z_positive, z_negatives, positive_cam_diff, device, alpha=positive_loss_weight, b=negative_loss_weight)

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
            negative_cam_diffs = negative_cam_diffs.float().to(device)

            z_negatives = []
            z_anchor = model(anchor_img)
            z_positive = model(positive_img)
            for j in range(num_negative_samples):
                z_negatives.append(model(negative_imgs[j].float().to(device)))

            loss_val = loss_fn(cos_loss, batch_size, z_anchor, z_positive, z_negatives, positive_cam_diff, device, alpha=positive_loss_weight, b=negative_loss_weight)
            running_loss += loss_val.item()
        
        avg_loss_test = running_loss / (i+1)
        
        training_loss.append(avg_loss_train)
        testing_loss.append(avg_loss_test)
        print(f'FOR EPOCH {epoch+1}/{num_epochs}, average training loss: {avg_loss_train}, average testing loss: {avg_loss_test}')
        if (epoch+1) % 5 == 0:
            torch.save({
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optim.state_dict(), 
                'scheduler_state_dict': scheduler.state_dict()
            }, f'{logs_path}/checkpoint_epoch{epoch}_a{positive_loss_weight}_b{negative_loss_weight}.pt')


    np.save(f'{logs_path}/training_loss.npy', np.array(training_loss))
    np.save(f'{logs_path}/testing_loss.npy', np.array(testing_loss))

    plt.figure()
    plt.plot(np.array(training_loss))
    plt.title('Training loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.savefig(str(f'{figs_path}/training_loss.png'))

    # training accuracy 
    plt.figure()
    plt.plot(np.array(testing_loss))
    plt.title('Testing Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Testing Loss')
    plt.savefig(str(f'{figs_path}/testing_loss.png'))

    print('yay')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--lr', default=1., type=float)
    parser.add_argument('--positive_loss', default=0.1, type=float )
    parser.add_argument('--negative_loss', default=0.1, type=float)
    parser.add_argument('--num_neg', default=10, type=int)
    parser.add_argument('--latent_size', default=128, type=int)
    args = parser.parse_args()
    train(args)