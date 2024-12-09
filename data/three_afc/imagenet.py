import random
from os.path import join
from torch import Tensor
from torchvision.datasets import ImageFolder

from data.constants import IMAGENET_1K_CLASS_ID_TO_LABEL, RESTRICTED_RANGES, \
    RESTRICTED_RANGES_TO_LABEL, BALANCED_RANGES, BALANCED_RANGES_TO_LABEL, \
    COARSE_IN, IMAGENET_CLASSNAMES


class ImageNet(ImageFolder):

    def __init__(self, data_dir, transforms_fn):
        super().__init__(data_dir, transforms_fn)
        self.class_names = IMAGENET_CLASSNAMES

    def class2label(self, cls):
        """Get class names from IDs."""

        if isinstance(cls, Tensor):
            cls = cls.tolist()
        elif not isinstance(cls, (list, tuple)):
            cls = [cls]
        labels = [IMAGENET_1K_CLASS_ID_TO_LABEL[cl] for cl in cls]
        return labels


class ImageNetTriplets(ImageNet):

    def __init__(self, data_dir, transforms_fn, verbose=False):

        super().__init__(data_dir, transforms_fn)
        assert data_dir.endswith('/val/') or data_dir.endswith('/val')
        self.verbose = verbose

    def __getitem__(self, idx):

        cls = random.sample(list(range(1000)), 2)  # Sample 2 classes (out of 1k).
        # print(cls)
        assert cls[0] != cls[1]
        lab = random.choice([0, 1, 2])  # Random position for the different class.
        classes = [cls[0] for _ in range(3)]
        classes[lab] = cls[1]
        if self.verbose:
            print(f'expected={self.class2label(classes)}')

        # Choose two images from the first class, one from the second one.
        imgs_idx = [random.randint(0, 49) + _cl * 50 for _cl in classes]
        outputs = [super(ImageNetTriplets, self).__getitem__(_idx) for _idx in imgs_idx]
        imgs = [a['pixel_values'].squeeze(0) for a, _ in outputs]
        if self.verbose:
            print(f'sampled={self.class2label([b for _, b in outputs])}')

        return imgs + [lab, idx]


class RestrictedImageNetTriplets(ImageNet):

    def __init__(self, data_dir, transforms_fn, verbose=False):

        super().__init__(data_dir, transforms_fn)
        assert data_dir.endswith('/val/') or data_dir.endswith('/val')
        self.verbose = verbose
        self.restricted_ranges = RESTRICTED_RANGES
        self.class_names = RESTRICTED_RANGES_TO_LABEL
        self.n_cls = len(self.restricted_ranges)
        self.is_range = True
        self.return_indiv_labels = False

    def coarse2label(self, cls):
        """Get coarse class names from IDs."""

        if isinstance(cls, Tensor):
            cls = cls.tolist()
        elif not isinstance(cls, (list, tuple)):
            cls = [cls]
        labels = [self.class_names[cl] for cl in cls]
        return labels

    def __getitem__(self, idx):

        cls = random.sample(list(range(self.n_cls)), 2)  # Sample 2 classes.
        # print(cls)
        assert cls[0] != cls[1]
        lab = random.choice([0, 1, 2])  # Random position for the different class.
        coarse_classes = [cls[0] for _ in range(3)]
        coarse_classes[lab] = cls[1]
        classes = []
        for _cl in coarse_classes:
            if self.is_range:
                ext_classes = list(range(self.restricted_ranges[_cl][0], self.restricted_ranges[_cl][1] + 1))
            else:
                ext_classes = list(self.restricted_ranges[_cl])
            classes.append(random.choice(ext_classes))
        if self.verbose:
            print(f'coarse_classes={self.coarse2label(coarse_classes)}')
            print(f'expected={self.class2label(classes)}')

        # Choose two images from the first class, one from the second one.
        imgs_idx = [random.randint(0, 49) + _cl * 50 for _cl in classes]
        outputs = [super(RestrictedImageNetTriplets, self).__getitem__(_idx) for _idx in imgs_idx]
        imgs = [a for a, _ in outputs]
        if self.verbose:
            print(f'sampled={self.class2label([b for _, b in outputs])}')

        if self.return_indiv_labels:
            return imgs + [lab, idx] + [self.coarse2label(coarse_classes)]
        return imgs + [lab, idx]


class BalRestrictedImageNetTriplets(RestrictedImageNetTriplets):

    def __init__(self, data_dir, transforms_fn, verbose=False):
        super().__init__(data_dir, transforms_fn, verbose)
        self.restricted_ranges = BALANCED_RANGES
        self.class_names = BALANCED_RANGES_TO_LABEL
        self.n_cls = len(self.restricted_ranges)


class HardImageNetTriplets(RestrictedImageNetTriplets):

    def __init__(self, data_dir, split, transforms_fn, verbose=False):
        super().__init__(join('/', 'imagenet/val'), transforms_fn, verbose)
        assert split == 'val'
        self.restricted_ranges = list(COARSE_IN.values())
        self.class_names = list(COARSE_IN.keys())
        self.n_cls = len(self.restricted_ranges)
        self.is_range = False
