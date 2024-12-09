from os.path import join

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_DIRS = {
    'distorted_images': 'distorted_images',
    'reference_images': 'reference_images',
    'test_reference_list': 'test_reference_list.txt',
    'train_reference_list': 'train_reference_list.txt',
    'val_reference_list': 'val_reference_list.txt',
}


class PieApp2AFCDataset(Dataset):
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

                if labels.iloc[idx][preference_col_name] > 0.6 or labels.iloc[idx][preference_col_name] < 0.4:

                    if (self.split == 'test' and (
                            labels.iloc[idx]['distorted image A'] == labels.iloc[idx]['ref. image']
                            or labels.iloc[idx]['distorted image B'] == labels.iloc[idx]['ref. image'])):
                        continue  # Skip when the reference is also one of the alternatives.

                    dataset.append(
                        {'ref_img': join(self.data_dir, 'reference_images', self.split, labels.iloc[idx]['ref. image']),
                         'img_A': join(self.data_dir, 'distorted_images', self.split, image,
                                       labels.iloc[idx]['distorted image A']),
                         'img_B': join(self.data_dir, 'distorted_images', self.split, image,
                                       labels.iloc[idx]['distorted image B']),
                         'label': 0 if labels.iloc[idx][preference_col_name] > 0.5 else 1,
                         'score': labels.iloc[idx][preference_col_name]})
        return dataset

    def __getitem__(self, idx):
        data = self.dataset[idx]
        imgs = [data['ref_img'],  # ref
                data['img_A'],  # left
                data['img_B']  # right
                ]
        p = data['label']

        if self.image_processor:
            imgs = [self.image_processor(Image.open(_img)) for _img in imgs]
        return *imgs, p, idx

    def __len__(self):
        return len(self.dataset)
