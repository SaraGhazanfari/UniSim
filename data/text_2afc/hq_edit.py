from torch.utils.data import Dataset
from os.path import join
import json
from PIL import Image


class HQEdit(Dataset):
    
    name = 'hq-edit'

    def __init__(self, data_dir, img_prepr=None, text_prepr=None, split=None, 
                 task_type='text_2afc', instruct=None):
        self.data_dir = data_dir
        self.split = split if split == 'train' else 'test'
        self.dataset = self._load_dataset()
        self.img_preprocess = img_prepr
        self.text_preprocess = text_prepr
        self.task_type = task_type
        self.instruct = instruct

    def _load_dataset(self):
        with open(join(self.data_dir, 'HQ-Edit', f'{self.split}.json')) as file:
            raw_dataset = json.load(file)
        
        dataset = list()
        for data in raw_dataset:
            if data['metadata']['dataset'] == 'HQ-Edit-Diff':
                continue
            dataset.append(data)
        
        return dataset

    def __getitem__(self, idx):
        sample = self.dataset[idx // 2]
        imgs = sample['image']
        caps = [sample['input'], sample['output']]
        
        if self.instruct and self.task_type == 'text_2afc':
            caps = [self.instruct.format(cap1=caps[0], cap2=caps[1])]
        elif self.instruct:
            caps = [self.instruct.format(prompt=cap) for cap in caps]
            
        if self.text_preprocess is not None:
            caps = [self.text_preprocess(cap) for cap in caps]
            
        imgs = [Image.open(join(self.data_dir, _im)).convert('RGB')  for _im in imgs]
        if self.img_preprocess is not None:
            imgs = [self.img_preprocess(_im) for _im in imgs]
            
        
        idx = idx % 2
        if self.task_type == 'text_2afc':
            return imgs[idx], caps, idx
        
        return *imgs, idx, caps[idx]

    def __len__(self):
        return 2 * len(self.dataset)
    
    
    