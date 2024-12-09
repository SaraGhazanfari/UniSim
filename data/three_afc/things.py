import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_DIRS = {
    'things': 'THINGS/Images/',
    'things-triplets': 'THINGS/triplet_dataset/triplets_large_final_correctednc_correctedorder.csv',
    'things-imgpaths': 'THINGS/things_first_imgs_idx.txt',
}
class THINGSDataset(Dataset):
    def __init__(self, data_dir, image_processor, n_triplets_max=-1):
        self.root_dir = os.path.join(data_dir, DEFAULT_DIRS['things'])
        img_file = os.path.join(data_dir, DEFAULT_DIRS['things-imgpaths'])
        triplets_file = os.path.join(data_dir, DEFAULT_DIRS['things-triplets'])
        with open(img_file, 'r') as f:
            self.img_paths = f.readlines()  # Paths to the example 1854 images.
            self.img_paths = [c.replace('\n', '') for c in self.img_paths]
            assert len(self.img_paths) == 1854
        triplets = pd.read_csv(triplets_file)
        triplets_data = triplets.values.squeeze().tolist()[:n_triplets_max]
        self.triplets = [c.split('\t')[:4] for c in triplets_data]
        self.preprocess_fn = image_processor

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        idx0, idx1, idx2, lab = [int(c) - 1 for c in self.triplets[idx]]

        img_0 = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.img_paths[idx0])))
        img_1 = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.img_paths[idx1])))
        img_2 = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.img_paths[idx2])))

        return img_0, img_1, img_2, lab, idx
