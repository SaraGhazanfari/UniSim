import random
from os.path import join
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os



DEFAULT_DIRS = {
    'koniq-images': 'koniq10k/1024x768/',  # 512x384
    'koniq-distr': 'koniq10k/koniq10k_distributions_sets.csv',
    'koniq-indicators': 'koniq10k/koniq10k_indicators.csv',
}    

class KonIQ10k(Dataset):

    name = 'koniq10k'

    def __init__(self, split, attribute, data_dir, transform_fn=None):

        self.split = 'training' if split == 'train' else split
        self.attribute = attribute
        assert attribute in ('brightness', 'contrast', 'colorfulness', 'sharpness', 'quality')
        
        self.data_dir = join(data_dir, DEFAULT_DIRS['koniq-images'])
        self.distr_sets = pd.read_csv(join(data_dir, DEFAULT_DIRS['koniq-distr']))
        self.distr_sets = self.distr_sets[self.distr_sets['set'] == self.split
                                          ].reset_index(drop=True)
        self.indicators = pd.read_csv(join(data_dir, DEFAULT_DIRS['koniq-indicators']))
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.distr_sets)
    
    def __getitem__(self, idx):

        img_name = self.distr_sets.loc[idx, 'image_name']
        img = Image.open(join(self.data_dir, img_name))
        if self.transform_fn:
            img = self.transform_fn(img)
        if self.attribute == 'quality':
            score = self.distr_sets.loc[idx, 'MOS']
        else:
            score = self.indicators[self.indicators['image_id'] == int(img_name.replace('.jpg', ''))
                                    ][self.attribute].item()

        return img, score


class KonIQ10kPairs(KonIQ10k):

    def __init__(self, split, attribute, data_dir, transform_fn=None, 
                 sampling_mode='rand-all', verbose=False, load_lib='pil'):
        
        super().__init__(split, attribute, data_dir, transform_fn)
        self.sampling_mode = sampling_mode
        self.verbose = verbose
        self.load_lib = load_lib

    def get_score(self, idx, name):
        if self.attribute == 'quality':
            return self.distr_sets.loc[idx, 'MOS']
        else:
            return self.indicators[self.indicators['image_id'] == int(name.replace(
                '.jpg', ''))][self.attribute].item()

    def __getitem__(self, idx):
        if self.sampling_mode == 'rand-all':
            other_idx = idx  # Second image.
            while other_idx == idx:
                other_idx = random.randint(0, len(self.distr_sets) - 1)
            img_names = [self.distr_sets.loc[idx, 'image_name'],
                         self.distr_sets.loc[other_idx, 'image_name']]
            imgs = [join(self.data_dir, _n) for _n in img_names] 
            if self.transform_fn:
                imgs = [self.transform_fn(Image.open(_img)) for _img in imgs]
            scores = [self.get_score(idx, img_names[0]),
                      self.get_score(other_idx, img_names[1])]
            sh_idx = [0, 1]
            random.shuffle(sh_idx)
            imgs = [imgs[i] for i in sh_idx]
            scores = [scores[i] for i in sh_idx]
            lab = int(scores[0] < scores[1])
            if self.verbose:
                print(img_names)

            return *imgs, lab, scores
                
    
# if __name__ == "__main__":
#     ds = KonIQ10kPairs(split='train', attribute='quality', data_dir='/vast/sg7457/uni_data')
#     print(len(ds))
