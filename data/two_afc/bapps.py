import os.path

import numpy as np
from PIL import Image

DEFAULT_DIRS = {'train': ['cnn', 'mix', 'traditional'],
                'val': ['cnn', 'color', 'deblur', 'frameinterp', 'superres', 'traditional']}


def make_dataset(base_path, part, ratio, split):

    images = []

    for dir in DEFAULT_DIRS[split]:
        dir = os.path.join(base_path, dir, part)
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        count_ratio = int(len(os.listdir(dir)) * ratio)
        for count, file in enumerate(sorted(os.listdir(dir))):
            path = os.path.join(dir, file)
            images.append(path)
            if count >= count_ratio:
                break
    return images


class BAPPSDataset:

    name = 'bapps'

    def __init__(self, root_dir: str, split: str = "val", load_size: int = 224, image_processor= None, ratio=1.0):
        split = split if split == 'train' else 'val'
        self.roots = os.path.join(root_dir, 'bapps', split)
        self.load_size = load_size
        self.image_processor = image_processor

        # image directory
        self.ref_paths = make_dataset(self.roots, 'ref', ratio=ratio, split=split)
        self.ref_paths = sorted(self.ref_paths)
        
        self.p0_paths = make_dataset(base_path=self.roots, part='p0', 
                                     ratio=ratio, split=split)
        self.p0_paths = sorted(self.p0_paths)


        self.p1_paths = make_dataset(self.roots, part='p1', ratio=ratio, split=split)
        self.p1_paths = sorted(self.p1_paths)

        # judgement directory
        self.judge_paths = make_dataset(self.roots, part='judge', ratio=ratio, split=split)
        self.judge_paths = sorted(self.judge_paths)

    def __getitem__(self, index):
        imgs = [self.ref_paths[index], self.p0_paths[index], self.p1_paths[index]]
        
        if self.image_processor:
            imgs = [Image.open(img).convert('RGB') for img in imgs]
            #print(imgs[0].size)
            imgs = [self.image_processor(img) for img in imgs]
            #print(imgs[0].shape)
            
        p = float(round(np.load(self.judge_paths[index])[0])) #.reshape((1, 1, 1,))  # [0,1]
        return *imgs, p, index

    def __len__(self):
        return len(self.p0_paths)