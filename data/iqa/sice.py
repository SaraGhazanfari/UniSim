import os
import time

from PIL import Image
from torch.utils.data import Dataset

DEFAULT_DIRS = {
    'sice-pairs-p1': 'sice/Dataset_Part1/',
    'sice-pairs-p2': 'sice/Dataset_Part2/',
}

def resize_image_by_scale(img, scale):
    width, height = img.size

    # Compute new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    return resized_img

def calculate_brightness(image_path):
    import numpy as np
    """Calculate the average brightness of an image."""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)  # Convert to a NumPy array
    brightness = img_array.mean()  # Calculate the mean pixel value
    return int(brightness)


class SICEPairs(Dataset):

    def __init__(self, data_dir, transform_fn=None, sampling_mode='rand-all',
                 split='train', verbose=False):

        self.data_dir = os.path.join(data_dir, 'sice')
        self.split = split if split == 'train' else 'test'
        annotation_file = os.path.join(self.data_dir, f'sice_{self.split}.json')

        self.l_imgs = self._get_dataset(
            self._get_images(['Dataset_Part1', 'Dataset_Part2']),
            annotation_file=annotation_file)

        self.transform_fn = transform_fn
        self.sampling_mode = sampling_mode
        self.verbose = verbose

    def _get_images(self, dirs):
        l_imgs = list()
        for dir in dirs:
            if dir == 'Label':
                continue
            l_imgs.extend([os.path.join(self.data_dir, dir,
                                        img_path) for img_path in os.listdir(os.path.join(self.data_dir, dir))])

        if self.split == 'train':
            return l_imgs[int(0.1 * len(l_imgs)):]
        else:
            return l_imgs[:int(0.1 * len(l_imgs))]

    def _get_dataset(self, l_imgs, annotation_file):

        datasets = list()
        startt = time.time()
        for img_idx, root_dir_per_image in enumerate(l_imgs):
            print(f'{img_idx} / {len(l_imgs)}, time: {time.time() - startt}', flush=True)
            available_imgs = [os.path.join(root_dir_per_image, img_path) for img_path in os.listdir(root_dir_per_image)]
            label_path = root_dir_per_image.split('/')
            label_path.insert(-2, 'Label')
            label_path[-1] += 'JPG'

            ranking = [calculate_brightness(img_path) for img_path in available_imgs]
            for idx in range(len(available_imgs) - 1):
                for alt_idx in range(idx, len(available_imgs)):
                    if ranking[idx] == ranking[alt_idx]:
                        continue
                    datasets.append({'imgs': [available_imgs[idx], available_imgs[alt_idx]],
                                     'ranks': [ranking[idx], ranking[alt_idx]]})

        return datasets

    def __len__(self):
        return len(self.l_imgs)

    def __getitem__(self, idx):
        imgs, scores = self.l_imgs[idx]['imgs'], self.l_imgs[idx]['ranks']
        imgs = [resize_image_by_scale(Image.open(_img).convert('RGB'), scale=0.2) for _img in imgs]
        lab = 0 if scores[0] > scores[1] else 1
        if self.verbose:
            print(imgs)
        if self.transform_fn:
            imgs = [self.transform_fn(_img) for _img in imgs]

        return *imgs, lab, scores

