import random
from os.path import join
from torch.utils.data import Dataset
import csv, ast
from PIL import Image


class Polaris(Dataset):

    name = 'polaris'

    def __init__(self, data_dir, img_prepr=None, text_prepr=None, split=None, instruct=None):
        
        self.split = split
        self.data_dir = join(data_dir, 'polaris')
        dataset = self._load_dataset()
        self.dataset = self._filter_dataset(dataset)
        self.img_preprocess = img_prepr
        self.text_preprocess = text_prepr
        self.instruct = instruct
    
    def _filter_dataset(self, dataset):
        new_dataset = list()
        for sample in dataset:
            img, ref_cap, cand_cap, human_score = sample[0], ast.literal_eval(sample[1]), sample[2], sample[3]
            for r_cap in ref_cap:
                new_dataset.append({
                    'img': img,
                    'ref_cap': r_cap,
                    'cand_cap': cand_cap,
                    'human_score': human_score
                })
                
        return new_dataset
            
            
    def _load_dataset(self):
        dataset = list()
        with open(join(self.data_dir, f'polaris_{self.split}.csv'), mode='r') as file:
            csv_reader = csv.reader(file)
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    continue
                processed_ref_cap_list = [ref_cap.lower().replace('.', '').replace('\n', '') for ref_cap in ast.literal_eval(row[1])]
                cand_cap = row[2].lower().replace('.', '').replace('\n', '')
                if float(row[-1]) <= 0.5 and cand_cap not in processed_ref_cap_list:
                    dataset.append(row)
        return dataset
            
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img, ref_cap, cand_cap = sample['img'], sample['ref_cap'], sample['cand_cap']
        img = join(self.data_dir, 'images', img)
        caps = [ref_cap, cand_cap]
        
        label = random.randint(0, 1)  
        if label == 1:
            caps = caps[::-1]
            
        if self.instruct:
            caps = [self.instruct.format(cap1=caps[0], cap2=caps[1])]
            
        if self.text_preprocess is not None:
            caps = [self.text_preprocess(cap) for cap in caps] 

        if self.img_preprocess is not None:
            img = Image.open(join(self.data_dir, 'images', img)).convert('RGB')
            # img = Image.open(join(self.data_dir, 'images', img))
            img = self.img_preprocess(img)
        
        return img, *caps, label

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    ds = Polaris(data_dir='/vast/sg7457/uni_data', split='test')
    print(len(ds))
    for i in range(10):
        print(ds.__getitem__(i))