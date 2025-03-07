import random
from os.path import join
from torch.utils.data import Dataset
import json
from PIL import Image
import os

DEFAULT_DIRS = {
    'annotation': 'data/annotations.json',  
    'val': 'image',
}


class AMBERDataset(Dataset):
    
    def __init__(self, data_dir, img_prepr=None, text_prepr=None, split=None, instruct=None, n_text=2):
        self.split = split if split == 'train' else 'val'
        self.data_dir = join(data_dir, 'AMBER')
        self.img_preprocess = img_prepr
        self.text_preprocess = text_prepr
        self.instruct = instruct
        self.n_text = n_text
        self.dataset = self._load_data()
        
    def _load_data(self):
        with open(join(self.data_dir, DEFAULT_DIRS['annotation'])) as file:
                annotations = json.load(file)
                
        dataset = list()
            
        for ann in annotations:
            if 'hallu' not in ann or len(ann['hallu']) != 5:
                continue
            for truth in ann['truth']:
                dataset.append({
                    'image': join(DEFAULT_DIRS[self.split], f'AMBER_{ann["id"]}.jpg'),
                    'truth': f'there is {truth} in this image',
                    'candidates': [f'there is {hallu} in this image' for hallu in ann['hallu'][:self.n_text]]
                    
                })
                
        return dataset

    def __getitem__(self, index):
        data = self.dataset[index]
        img = join(self.data_dir, data['image'])
        
        if self.img_preprocess is not None:
            img = self.img_preprocess(Image.open(img).convert('RGB'))
            
        label = random.randint(0, self.n_text-1)
        data['candidates'].insert(label, data['truth'])
        texts = data['candidates']
        if self.text_preprocess is not None:
            texts = self.text_preprocess(texts)
        
        return img, texts, label
    
    def __len__(self):
        return len(self.dataset)
    
