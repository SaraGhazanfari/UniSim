import json
import random
from os.path import join

from PIL import Image
from torch.utils.data import Dataset


class MagicBrush(Dataset):
    name = 'magic-brush'

    def __init__(self, data_dir, img_prepr=None, text_prepr=None, split='train', instruct=None):
        self.data_dir = data_dir
        self.split = split if split == 'train' else 'test'
        self.dataset = self._load_dataset()
        self.img_preprocess = img_prepr
        self.text_preprocess = text_prepr
        self.instruct = instruct

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        imgs = list()
        for _im in sample['image']:
            imgs.append(_im.split('/'))
            imgs[-1][1] = self.split
            imgs[-1] = '/'.join(imgs[-1])
            imgs[-1] = join(self.data_dir, imgs[-1])

        caps = sample['conversations'][1]['value']
        if self.instruct:
            caps = self.instruct.format(prompt=caps)

        if self.text_preprocess is not None:
            caps = self.text_preprocess(caps)

        if self.img_preprocess is not None:
            imgs = [self.img_preprocess(Image.open(_im).convert('RGB')) for _im in imgs]
            # imgs = [self.img_preprocess(Image.open(_im).convert('RGB')) for _im in imgs]

        label = random.randint(0, 1)  # 0 or 1
        if label == 0:
            imgs = imgs[::-1]  # Reverse the image order if label is 0

        return *imgs, label, caps

    def __len__(self):
        return len(self.dataset)

    def _load_dataset(self):
        with open(join(self.data_dir, 'MagicBrush', f'{self.split}.json')) as file:
            raw_dataset = json.load(file)

        dataset = list()
        for data in raw_dataset:
            if data['metadata']['dataset'] == 'MagicBrush-Diff' or len(data['image']) > 2:
                continue
            dataset.append(data)

        return dataset
