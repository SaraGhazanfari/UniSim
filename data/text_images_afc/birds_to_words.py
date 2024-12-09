from torch.utils.data import Dataset
import json, random
from os.path import join
from PIL import Image

class BirdsToWords(Dataset):
    def __init__(self, data_dir, img_prepr=None, text_prepr=None, split=None, instruct=None):
        data_dir = join(data_dir, 'Birds-to-Words')
        self.image_dir = join(data_dir, 'full', 'images')
        self.split = split
        self.img_preprocess = img_prepr
        self.text_preprocess = text_prepr
        self.instruct = instruct
        self.annotations = self._read_ann(data_dir)
        
    def _read_train_test_split(self, data_dir, split): 
        data_split = {'train': [], 'test': []}
        import csv

        with open(join(data_dir, 'birds-to-words-v1.0.tsv'), newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                if row[8] == 'train':
                    data_split['train'].append(row[10])
                else:
                    data_split['test'].append(row[10])
        return data_split
    
    def _read_ann(self, data_dir): 
        data_split = self._read_train_test_split(data_dir, self.split)
        out_annotations = list()
        with open(join(data_dir, 'full', 'full.json')) as file:
            annotations = json.load(file)['data']
        for ann in annotations:
            if ann['response'] in data_split[self.split]:
                out_annotations.append(ann)
        
        return self._extend_samples(out_annotations)
            
    def _extend_samples(self, annotations):
        # The captions have multiple sentences. this way we'll have separated samples for each single sentence
 
        new_annotations = list()
        for ann in annotations:
            if 'there are no differences' in ann['response']:
                continue
            all_caps = ann['response'].split('.')
            for pre_cap in all_caps:
                cap_list = pre_cap.split('animal')
                for cap in cap_list:
                    if cap.startswith('1') or cap.startswith('2'):
                        new_ann = ann.copy()
                        new_ann['label'] = int(cap[0])
                        new_ann['response'] = f'this animal {cap[1:].strip()}'
                        new_annotations.append(new_ann)
        return new_annotations

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        imgs = sample['task_instance']['images_path']
        imgs = [Image.open(join(self.image_dir, _img)) for _img in imgs]
        caps = sample['response']
        label = sample['label'] - 1
        if self.instruct:
            prompt = self.instruct.format(prompt=prompt)
        if self.text_preprocess is not None:
            caps = self.text_preprocess(caps)

        if self.img_preprocess is not None:
            imgs = [self.img_preprocess(_im) for _im in imgs]
        return *imgs, label, caps

    def __len__(self):
        return len(self.annotations)
    
if __name__ == "__main__":
    ds = BirdsToWords(data_dir='/vast/sg7457/uni_data', split='train')
    print(len(ds))
    print(ds.__getitem__(0))

    