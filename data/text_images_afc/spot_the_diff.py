from torch.utils.data import Dataset
import json, random
from os.path import join
from PIL import Image

class SpotTheDiff(Dataset):
    # not sure about this datasets, the captions sometimes are not good, in a sense that the two images can be good,
    # maybe we can add curate this a little bit
    def __init__(self, data_dir, img_prepr=None, text_prepr=None, split=None, test_size=1397):
        data_dir = join(data_dir, 'spot-the-diff')
        self.image_dir = join(data_dir, 'full', 'images')
        self.split = split
        self.img_preprocess = img_prepr
        self.text_preprocess = text_prepr
        self.annotations = self._read_ann(data_dir, test_size)
        
    def _read_ann(self, data_dir, test_size): 
        # the split is not very obvious, there are some json files here https://github.com/harsh19/spot-the-diff/tree/master/data/annotations
        # but i couldn't match it with the images. based on the captions i figured out that after 1397 is the training data, we can have our own
        with open(join(data_dir, 'full', 'full.json')) as file:
            annotations = json.load(file)['data']
        if self.split == 'train':
            annotations = annotations[test_size:]
        else:
            annotations = annotations[:test_size]
        
        return self._extend_samples(annotations)
            
    def _extend_samples(self, annotations):
        # The captions have multiple sentences. this way we'll have separated samples for each single sentence
        # e.g., "there is a car driving that wasn t there before. there are now three people standing near the moving car. 
        # there are now two people at the bottom left side instead of the three that were there before"   
        new_annotations = list()
        for ann in annotations:
            if 'there are no differences' in ann['response']:
                continue
            all_caps = ann['response'].split('.')
            for cap in all_caps:
                new_ann = ann.copy()
                new_ann['response'] = cap
                new_annotations.append(new_ann)
        return new_annotations

    def __getitem__(self, idx):

        imgs = self.annotations[idx]['task_instance']['images_path']
        imgs = [Image.open(join(self.image_dir, _img)) for _img in imgs]
        caps = self.annotations[idx]['response']

        if self.text_preprocess is not None:
            caps = self.text_preprocess(caps)

        if self.img_preprocess is not None:
            imgs = [self.img_preprocess(_im) for _im in imgs]
            
        label = random.randint(0, 1)  # 0 or 1
        if label == 0:
            imgs = imgs[::-1]  # Reverse the image order if label is 0

        return *imgs, label, caps

    def __len__(self):
        return len(self.annotations)
    