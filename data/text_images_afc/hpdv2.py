import json
import random
from os.path import join

from PIL import Image
from torch.utils.data import Dataset


class HPDv2Pairs(Dataset):
    name = 'hpdv2'

    def __init__(self, data_dir, img_prepr=None, text_prepr=None, split=None,
                 verbose=False, instruct=None, sampling_mode='rand-all', n_imgs=2,
                 min_dist=0):

        self.corrupted_imgs = ['00006811.jpg', '00168743.jpg', '00331911.jpg', '00310087.jpg']
        self.data_dir = join(data_dir, 'hpdv2')
        self.split = split if split == 'train' else 'test'
        self.dataset = self._load_dataset()
        self.img_preprocess = img_prepr
        self.text_preprocess = text_prepr
        self.verbose = verbose
        self.instruct = instruct
        self.sampling_mode = sampling_mode
        self.n_imgs = n_imgs
        self.min_dist = min_dist

    def _load_dataset(self):

        with open(join(self.data_dir, self.split + '.json')) as file:
            raw_dataset = json.load(file)
        dataset = list()

        for data in raw_dataset:
            not_save = False
            for img in data['image_path']:
                if img in self.corrupted_imgs:
                    not_save = True
                    break
            if not not_save:
                dataset.append(data)

        return dataset

    def __getitem__(self, idx):

        data = self.dataset[idx]
        prompt = data['prompt']
        if self.instruct:
            prompt = self.instruct.format(prompt=prompt)
        if self.text_preprocess is not None:
            prompt = self.text_preprocess(prompt)

        if self.split == 'test':  # 'train' and 'test' sets have different structure.
            if self.sampling_mode == 'rand-all':  # Sample `n_imgs` random images with same prompt.
                ranks = random.sample(data['rank'], self.n_imgs)
                if self.n_imgs == 2:
                    assert ranks[0] != ranks[1]
                    lab = int(ranks[0] > ranks[1])
                else:
                    assert len(ranks) == len(set(ranks))
                    random.shuffle(ranks)
                    lab = ranks.index(min(ranks))
                imgs = [data['image_path'][data['rank'].index(_rank)] for _rank in ranks]
            elif self.sampling_mode == 'easy':
                assert self.n_imgs == 2
                ranks = [min(data['rank']), max(data['rank'])]  # Sample images with best and worst rank.
                assert ranks[0] != ranks[1]
                random.shuffle(ranks)
                lab = int(ranks[0] > ranks[1])
                imgs = [data['image_path'][data['rank'].index(_rank)] for _rank in ranks]
            elif self.sampling_mode == 'min-dist':
                assert self.n_imgs == 2
                assert self.min_dist > 0
                rank0 = random.choice(data['rank'])  # Sample images with best and worst rank.
                rem_ranks = [_rank for _rank in data['rank'] if abs(rank0 - _rank) >= self.min_dist]
                rank1 = random.choice(rem_ranks)
                ranks = [rank0, rank1]
                assert ranks[0] != ranks[1]
                random.shuffle(ranks)
                lab = int(ranks[0] > ranks[1])
                imgs = [data['image_path'][data['rank'].index(_rank)] for _rank in ranks]
        else:
            lab = data['human_preference'][1]
            imgs = data['image_path']
            if lab == 1:
                lab = random.randint(0, 1)
                if lab == 0:
                    imgs = imgs[::-1]

        imgs = [join(self.data_dir, self.split, _im) for _im in imgs]
        if self.img_preprocess is not None:
            imgs = [Image.open(join(self.data_dir, self.split, _im)).convert('RGB') for _im in imgs]
            imgs = [self.img_preprocess(_im) for _im in imgs]
        
        if self.n_imgs > 2:
            return imgs, lab, prompt

        return *imgs, lab, prompt

    def __len__(self):
        if self.split == 'train':
            return len(self.dataset) - len(self.corrupted_imgs)
        return len(self.dataset)
