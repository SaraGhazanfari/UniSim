import random
from os.path import join
from torch.utils.data import Dataset
import json
from PIL import Image


DEFAULT_DIRS = {
    'images': '/coco/{split}2014/',  
    'val': 'foil/foilv1.0_test_2017.json',
    'train': 'foil/foilv1.0_train_2017.json'
}

class Foil(Dataset):

    def __init__(self, data_dir, img_prepr=None, text_prepr=None, split=None, instruct=None):
        self.split = split if split == 'train' else 'val'
        self.data_dir = data_dir
        self.annotation = self._read_annotation()
        self.img_preprocess = img_prepr
        self.text_preprocess = text_prepr
        self.instruct = instruct
    
    def _read_annotation(self):
        with open(join(self.data_dir, DEFAULT_DIRS[self.split])) as file:
            annotations = json.load(file)['annotations']
            
        self.annotations = dict()
        id_set = set()
        for ann in annotations:   
            id = ann['id']
            id_set.add(id)
            ann['image_id'] = f"COCO_{self.split}2014_{ann['image_id']:012d}.jpg" 
            self.annotations[id] = self.annotations.get(id, list())
            self.annotations[id].append(ann)
        self.id_list = list(id_set)   
                     
    def __getitem__(self, idx):
        samples = self.annotations[self.id_list[idx]]
        caps = [sam['caption'] for sam in samples]
        label = 0 if not samples[0]['foil'] else 1
        img = Image.open(join(DEFAULT_DIRS['images'].format(split=self.split), 
                              samples[0]['image_id']))   
        if self.instruct:
            caps = [self.instruct.format(cap1=caps[0], cap2=caps[1])]
            
        if self.text_preprocess is not None:
            caps = [self.text_preprocess(cap) for cap in caps]
            
        if self.img_preprocess is not None:
            img = self.img_preprocess(img)

        return img, *caps, label

    def __len__(self):
        return len(self.id_list)
    