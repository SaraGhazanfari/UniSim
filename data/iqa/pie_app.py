from os.path import join
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import random


DEFAULT_DIRS = {
    'distorted_images': 'distorted_images',
    'reference_images': 'reference_images',
    'test_reference_list': 'test_reference_list.txt',
    'train_reference_list': 'train_reference_list.txt',
    'val_reference_list': 'val_reference_list.txt',
}


class PieAppIQADataset(Dataset):
    
    name = 'pie_app'

    def __init__(self, data_dir: str, split: str = "train", load_size: int = 224, image_processor=None):
        self.data_dir = join(data_dir, 'pie_app')
        self.split = split
        image_list = self._load_reference_list()
        self.dataset = self._create_dataset(image_list)
        self.image_processor = image_processor
    
    def _load_reference_list(self):
        dataset = list()
        with open(join(self.data_dir, DEFAULT_DIRS[f'{self.split}_reference_list'])) as file:
            for line in file.readlines():
                dataset.append(line.split('\n')[0].split('.png')[0])
        return dataset
    
    def _create_dataset(self, image_list):
        preference_col_name = 'processed preference for A' if self.split != 'test' else 'preference for A'
        dataset = list()
        for image in image_list:
            labels = pd.read_csv(join(self.data_dir, 'labels', self.split, f'{image}_pairwise_labels.csv'))
            labels = labels.rename(columns=lambda x: x.lstrip())
            for idx in range(len(labels)):

                if (self.split == 'test' and (
                    labels.iloc[idx]['distorted image A'] == labels.iloc[idx]['ref. image']
                    or labels.iloc[idx]['distorted image B'] == labels.iloc[idx]['ref. image'])):
                    continue  # Skip when the reference is also one of the alternatives.
                
                if labels.iloc[idx][preference_col_name] > 0.6 or labels.iloc[idx][preference_col_name] < 0.4:
                    dataset.append({'ref_img': join(self.data_dir, 'reference_images', self.split, labels.iloc[idx]['ref. image']),
                                    'img_A': join(self.data_dir, 'distorted_images', self.split, image, labels.iloc[idx]['distorted image A']),
                                    'img_B': join(self.data_dir, 'distorted_images', self.split, image, labels.iloc[idx]['distorted image B']),
                                    'label': 0 if labels.iloc[idx][preference_col_name] > 0.5 else 1,
                                    'score': labels.iloc[idx][preference_col_name]})
        return dataset
            
    def __getitem__(self, idx):
        data = self.dataset[idx // 2]
        idx = idx % 2
        if idx == 0:
            imgs = [Image.open(data['ref_img']), Image.open(data['img_A'])] 
        else:             
            imgs = [Image.open(data['ref_img']), Image.open(data['img_B'])] 
        
        if self.image_processor:
            imgs = [self.image_processor(_img) for _img in imgs]
        
        lab = random.randint(0, 1)
        if lab == 1:
            imgs = imgs[::-1]
        return *imgs, lab, 0
    
    def __len__(self):
        return len(self.dataset) * 2