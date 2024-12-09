import numpy as np
from torchvision.datasets import CIFAR100
import random


# From https://github.com/ryanchankh/cifar100coarse.
class CIFAR100Coarse(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        self.targets = coarse_labels[self.targets]

        # update classes
        self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
        

class CIFAR100CoarseTriplets(CIFAR100Coarse):
    """
    Creates triples of images for odd-one-out class. The images are from two classes,
    the label is the position of the image from the different class.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, verbose=False):
        super().__init__(root, train, transform, target_transform, download)
        
        self.unique_targets = np.unique(self.targets)
        self.class_indices = {
            _cl: np.argwhere(self.targets == _cl).squeeze() for _cl in self.unique_targets}
        self.verbose = verbose
        
    def __getitem__(self, idx):

        cls = np.random.choice(self.unique_targets, 2, replace=False)  # Sample 2 classes.
        assert cls[0] != cls[1]
        lab = random.choice([0, 1, 2])  # Random position for the different class.
        classes = [cls[0].item() for _ in range(3)]
        classes[lab] = cls[1].item()
        
        # Choose two images from the first class, one from the second one.
        imgs_idx = [np.random.choice(self.class_indices[_cl], 1, replace=False
                                     ).item() for _cl in classes]
        outputs = [super(CIFAR100CoarseTriplets, self).__getitem__(_idx) for _idx in imgs_idx]
        imgs = [a for a, _ in outputs]
        if self.verbose:
            for _, b in outputs:
                print(self.classes[b])

        return imgs + [lab, idx]
        
        