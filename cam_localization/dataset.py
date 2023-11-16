import glob
import torch 
import cv2 as cv
import numpy as np  
from pathlib import Path
from random import sample 
import json 
from torch.utils.data import Dataset 

class EndoscopicDataset(Dataset):

    def __init__(self, data_dir, train=True, rendered_transforms=None, styletransfer_transforms=None, num_negative_samples=10):
        
        if train:
            traj_dir = data_dir / 'cleared_data' / 'clustered_cam_poses.json'
            self.img_dir = data_dir / 'cleared_data' / 'images'
            
        else:
            traj_dir = data_dir / 'validation_data_dilated1' / 'clustered_cam_poses.json'
            self.img_dir = data_dir / 'validation_data_dilated1' / 'images'
        
        if not self.img_dir.exists():
            raise Exception('images directory does not exist')
        if not traj_dir.exists():
            raise Exception('cam_traj.txt does not exist.')
        
        self.style_imgs = glob.glob(str(self.img_dir / '*fake.png')) + glob.glob(str(self.img_dir / '*fake.jpg'))
        self.num_negative_samples = num_negative_samples
        self.rendered_transforms = rendered_transforms
        self.styletransfer_transforms = styletransfer_transforms

        # cam trajectory dictionary. 
        # key: name of image
        # value: dictionary with keys 'positive' and 'negative'
        # value of values: list of tuples with format (image_name, dtx, dty, dtz, dtheta)
        f = open(str(traj_dir))
        self.trajs = json.load(f)
        f.close()
    
    def find_rendered(self, key_name):
        filename = self.img_dir / (key_name[:-4] + '_real.png')
        return cv.imread(str(filename))
    
    def _sample(self, candidates, num_candidates):
        i = 0
        res_imgs, res_cam_diffs = [], []
        while i < num_candidates: 
            winner = sample(candidates, 1)[0]
            filename = self.img_dir / (winner[0][:-4] + '_real.png')
            if not filename.exists():
                continue 

            if num_candidates == 1:
                return cv.imread(str(filename)), winner[1:]
            else: 
                res_imgs.append(cv.imread(str(filename)))
                res_cam_diffs.append(winner[1:])
                i += 1
        return res_imgs, res_cam_diffs

    def __len__(self):
        return len(self.style_imgs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        filename = Path(self.style_imgs[idx])
        anchor_img = cv.imread(str(filename))

        key = filename.name[:-9] + '.png'
        positive_candidates = self.trajs[key]['positive']
        positive_img, positive_cam_diff = self._sample(positive_candidates, 1)

        negative_candidates = self.trajs[key]['negative']
        negative_imgs, negative_cam_diffs = self._sample(negative_candidates, self.num_negative_samples)

        if self.rendered_transforms:
            positive_img = self.rendered_transforms(positive_img)
            negative_imgs = [self.rendered_transforms(x) for x in negative_imgs]
        if self.styletransfer_transforms:
            anchor_img = self.styletransfer_transforms(anchor_img)

        return anchor_img, positive_img, torch.Tensor(positive_cam_diff), negative_imgs, torch.Tensor(negative_cam_diffs)
    