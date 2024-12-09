import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class NightDataset(Dataset):

    name = 'nights'

    def __init__(self, root_dir: str, split: str = "train", load_size: int = 224, image_processor: str = "DEFAULT",
                 #tokenizer=None, #prompt=None
                 ):
        self.root_dir = os.path.join(root_dir, 'nights')
        self.csv = pd.read_csv(os.path.join(self.root_dir, "data.csv"))
        self.csv = self.csv[self.csv['votes'] >= 6]  # Filter out triplets with less than 6 unanimous votes
        self.split = split
        #self.prompt = prompt
        self.load_size = load_size
        #self.tokenizer = tokenizer
        self.image_processor = image_processor
        #self.image_processor = image_processor
        if self.split == "train" or self.split == "val":
            self.csv = self.csv[self.csv["split"] == split]
        elif split == 'test':
            self.csv = self.csv[self.csv['split'] == 'test']
        elif split == 'test_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == True]
        elif split == 'test_no_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == False]
        else:
            raise ValueError(f'Invalid split: {split}')

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        p = self.csv.iloc[idx, 2].astype(np.float32)
        imgs = [os.path.join(self.root_dir, self.csv.iloc[idx, 4]), #ref
                os.path.join(self.root_dir, self.csv.iloc[idx, 5]),  # left
                os.path.join(self.root_dir, self.csv.iloc[idx, 6])  # right
                ] 
        if self.image_processor:
            imgs = [self.image_processor(Image.open(_img)) for _img in imgs]
        return *imgs, p, idx

