import os
import random
from os.path import join
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_DIRS = {
    'train_alt_imgs': 'Dist_Imgs',
    'train_ref_imgs': 'Train_Ref',
    'train_labels': 'Train_Label',
    'val_ref_imgs': 'Val_Ref',
    'val_alt_imgs': 'Val_Dist',
    'val_labels': 'val_label.txt'
}

class PiPal(Dataset):

    name = 'pipal'

    def __init__(self, data_dir, image_processor=None, split='train'):
        self.data_dir = join(data_dir, 'PIPAL')
        self.split = split if split == 'train' else 'val'
        self.image_processor = image_processor
        imgs = os.listdir(join(self.data_dir, DEFAULT_DIRS[f'{self.split}_ref_imgs']))
        imgs = [img.replace('.bmp', '') for img in imgs]
        self.dataset = self._filter_dataset(self._load_dataset(imgs))

    def _load_dataset(self, imgs):
        dataset = dict()
        for img in imgs:
            label_path = join(self.data_dir, DEFAULT_DIRS[f'{self.split}_labels'])
            label_path = join(label_path, f'{img}.txt') if self.split == 'train' else label_path
            with open(label_path) as file:
                dataset[img] = {'ref':[join(self.data_dir, DEFAULT_DIRS[f'{self.split}_ref_imgs'], f'{img}.bmp')],
                                       'alt':[]}
                for line in file.readlines():
                    alt_image, score = line.split(',')
                    if img in alt_image:
                        img_path = join(self.data_dir, DEFAULT_DIRS[f'{self.split}_alt_imgs'], alt_image)
                        if float(score) >= 1500:
                            dataset[img]['ref'].append(img_path)
                        else:
                            dataset[img]['alt'].append(img_path)
        return dataset
    
    def _filter_dataset(self, dataset):
        new_dataset = list()
        for _, value in dataset.items():
            for idx, ref in enumerate(value['ref']):
                for alt in value['alt']:
                    new_dataset.append({'ref_img': ref,
                                        'alt_img': alt})
                if idx > 3:
                    break

        return new_dataset
        
    def __getitem__(self, idx):
        data = self.dataset[idx]
        imgs = [Image.open(data['ref_img']), Image.open(data['alt_img'])] 

        if self.image_processor:
            imgs = [self.image_processor(_img) for _img in imgs]
        
        lab = random.randint(0, 1)
        if lab == 1:
            imgs = imgs[::-1]
        return *imgs, lab, 0
    
    def __len__(self):
        return len(self.dataset)
    
if __name__ == "__main__":
    ds = PiPal(data_dir='/vast/sg7457/uni_data', split='train')
    print(len(ds))
    for i in range(15):
        print(ds.__getitem__(i))
    