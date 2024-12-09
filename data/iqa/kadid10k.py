import os
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class KADID10k(Dataset):

    name = 'kadid10k'

    def __init__(self, root_dir, transform_fn, split):
        self.root_dir = root_dir
        assert split in ('train', 'test')
        self.split = split
        csv = pd.read_csv(os.path.join(root_dir, 'dmos.csv'))
        self.csv = self._get_data_split(csv, split)
        self.transform_fn = transform_fn
        if self.transform_fn is None:
            self.transform_fn = lambda x: x
        self.n_imgs = 72 if split == 'train' else 9
        self.n_corruptions = 25
        self.n_severities = 5

    def _get_data_split(self, csv, split):
        num_rows = 9000 #int(0.9 * len(csv))
        if split == 'train':
            return csv.iloc[:num_rows]
        else: 
            return csv.iloc[num_rows:]

    def __getitem__(self, idx):
        dist = os.path.join(self.root_dir, 'images', self.csv.loc[idx, 'dist_img'])
        ref = os.path.join(self.root_dir, 'images', self.csv.loc[idx, 'ref_img'])
        if self.transform_fn is not None:
            dist = Image.open(dist)
            ref = Image.open(ref)

        return dist, ref, self.csv.loc[idx, 'dmos'], self.csv.loc[idx, 'var']

    def __len__(self):
        return len(self.csv) #self.n_imgs


class KADID10kPairs(KADID10k):

    def __init__(self, root_dir, split, transform_fn=None, sampling_mode='sev-noref',
                 verbose=False):
        super().__init__(os.path.join(root_dir, 'kadid10k'), transform_fn, split)
        self.sampling_mode = sampling_mode
        self.verbose = verbose
        self.transform_fn = transform_fn

    def __getitem__(self, _):

        if self.sampling_mode in ['sev-noref', 'easy-sev-noref']:
            # Fixed image and corruption, different severities.
            img = random.randint(1, self.n_imgs)  # idx + 1
            if self.split == 'test':
                img += 72
            corr = random.randint(1, self.n_corruptions)
            if self.sampling_mode == 'sev-noref':
                sevs = random.sample(list(range(1, self.n_severities + 1)), 2)
            elif self.sampling_mode == 'easy-sev-noref':
                sevs = [1, 5]  # Lowest and highest severities.
            assert sevs[0] != sevs[1]
            random.shuffle(sevs)
            imgs = [f'I{img:02d}_{corr:02d}_{sev:02d}.png' for sev in sevs]
            lab = int(sevs[0] > sevs[1])  # 0 -> img0 is the better one (lower severity)
            if self.verbose:
                print(sevs, f'label={lab}')
            imgs = [os.path.join(self.root_dir, 'images', _im) for _im in imgs]
            
            if self.transform_fn is not None:
                imgs = [self.transform_fn(Image.open(_im)) for _im in imgs]

            return *imgs, lab, 0

        elif self.sampling_mode in ['corr-noref', ]:
            # Fixed image and severity, different corruptions.
            img = random.randint(1, self.n_imgs)
            corrs = random.sample(list(range(1, self.n_corruptions + 1)), 2)
            sev = random.randint(1, self.n_severities)
            assert corrs[0] != corrs[1]
            random.shuffle(corrs)
            imgs = [f'I{img:02d}_{corr:02d}_{sev:02d}.png' for corr in corrs]
            idx = [self.csv[self.csv['dist_img'] == _im].index for _im in imgs]
            dmos = [self.csv['dmos'][i].values.item() for i in idx]
            lab = int(dmos[0] < dmos[1])  # 0 -> img0 is the better one (higher mos)
            if self.verbose:
                print(f'corr={corrs}', f'sev={sev}', f'mos={dmos}', f'label={lab}')

            imgs = [os.path.join(self.root_dir, 'images', _im) for _im in imgs]
            if self.transform_fn is not None:
                imgs = [self.transform_fn(Image.open(_im)) for _im in imgs]

            return *imgs, lab, 0
