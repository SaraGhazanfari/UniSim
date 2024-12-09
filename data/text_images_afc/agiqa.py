import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import random


DEFAULT_DIRS = {
    'agiqa-image': 'agiqa-3k/images',
    'agiqa-info': 'agiqa-3k/data.csv',
}


class AGIQA3K(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations (data.csv).
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample (default: None).
        """

        root_dir = os.path.join(data_dir, DEFAULT_DIRS['agiqa-image'])
        csv_file = os.path.join(data_dir, DEFAULT_DIRS['agiqa-info'])
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.data_info)

    def __getitem__(self, idx):
        """Fetch a single data sample by index."""
        # Get image file name from the first column of the CSV
        img_name = self.data_info.iloc[idx, 0]
        
        # Construct the full path to the image
        img_path = os.path.join(self.root_dir, img_name)
        
        # Load the image using PIL
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        # Load the metadata for this sample
        prompt = self.data_info.iloc[idx, 1]
        adj1 = self.data_info.iloc[idx, 2]
        adj2 = self.data_info.iloc[idx, 3]
        style = self.data_info.iloc[idx, 4]
        
        # Load the normalized MOS (Mean Opinion Score) and STD (Standard Deviation)
        perception_mos = self.data_info.iloc[idx, 5]
        perception_std = self.data_info.iloc[idx, 6]
        alignment_mos = self.data_info.iloc[idx, 7]
        alignment_std = self.data_info.iloc[idx, 8]
        
        return image, prompt, perception_mos, alignment_mos
    

class AGIQA3KPairs(AGIQA3K):

    def __init__(self, data_dir, transform=None, metric='mos_align',
                 min_mos_quality=0, n_imgs=2, verbose=False, instruct=None, text_preprocess=None):
        super().__init__(data_dir, transform)

        self.prompts = self.data_info['prompt'].unique().tolist()
        self.metric = metric
        assert self.metric in ('mos_align', 'mos_quality')
        self.min_quality = min_mos_quality
        self.verbose = verbose
        self.n_imgs = n_imgs
        self.instruct = instruct
        self.text_preprocess = text_preprocess  

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.prompts)
    
    def __getitem__(self, idx):
        
        prompt = self.prompts[idx]
        indices = self.data_info[self.data_info['prompt'] == prompt].index.tolist()
        mos = self.data_info[self.data_info['prompt'] == prompt][self.metric].tolist()
        
        if self.instruct:
            prompt = self.instruct.format(prompt=prompt)
            
        if self.text_preprocess is not None:
            prompt = self.text_preprocess(prompt)
            
        indices = [i for i, m in zip(indices, mos) if m >= self.min_quality]
        indices = random.sample(indices, self.n_imgs)
        img_names = [self.data_info.iloc[i, 0] for i in indices]
        imgs = [os.path.join(self.root_dir, _im) for _im in img_names]
        if self.transform:
            imgs = [Image.open(_im) for _im in imgs]
            imgs = [self.transform(_im) for _im in imgs]
        mos = [self.data_info.loc[i, self.metric] for i in indices]
        #lab = int(mos[0] < mos[1])  # Image with higher score.
        lab = mos.index(max(mos))
        if self.verbose:
            print(mos)

        if self.n_imgs > 2:
            return imgs, lab, prompt
        return *imgs, lab, prompt