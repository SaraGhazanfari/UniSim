"""Adapted from https://github.com/zwx8981/LIQE/blob/main/utils.py"""

import torch
from torchvision.transforms import Compose, ToTensor, Normalize, \
    InterpolationMode, Resize


class AdaptiveResize(object):
    """Resize the input PIL Image to the given size adaptively.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, image_size=None):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation
        if image_size is not None:
            self.image_size = image_size
        else:
            self.image_size = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size

        if self.image_size is not None:
            if h < self.image_size or w < self.image_size:
                return Resize(self.image_size, self.interpolation)(img)

        if h < self.size or w < self.size:
            if (h <= 288 and w <= 288) or (min(h, w) <= 311):  # Not original: added to make it work with small images.
                return Resize(384, self.interpolation)(img)
            return img
        else:
            return Resize(self.size, self.interpolation)(img)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _preprocess2():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def liqe_prepr(
        I,
        preprocess=_preprocess2(),
        test=True,
        step=32,
        num_patch=15,
        ):

        init_size = I.size
        #print(init_size)
        I = preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 32
        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(
            0, 2, 3, 1, 4, 5).reshape(-1, n_channels, kernel_h, kernel_w)

        assert patches.size(0) >= num_patch, (init_size, I.shape, patches.shape, num_patch)
        #self.num_patch = np.minimum(patches.size(0), self.num_patch)
        if test:
            sel_step = patches.size(0) // num_patch
            sel = torch.zeros(num_patch)
            for i in range(num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(num_patch, ))
        patches = patches[sel, ...]

        return patches
