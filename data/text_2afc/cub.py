import os
import random

import torch

from torch.utils.data import Dataset
from torchvision import transforms


# Adapted from https://www.kaggle.com/code/coolerextreme/utils/notebook.
# This uses already pre-processed and saved as torch.Tensor image, at maximum
# resolution 256px for the test set.
class CUB_200_2011(Dataset):
    """
    If should_pad is True, need to also provide a pad_to_length. Padding also adds
    <START> and <END> tokens to captions.
    """
    
    def __init__(
            self, data_dir, split, d_image_size=256, transform=None, should_pad=False,
            pad_to_length=None, no_start_end=False, return_decoded=True):
        super().__init__()

        assert split in ('all', 'train_val', 'test')
        assert d_image_size in (64, 128, 256)
        if should_pad:
            assert pad_to_length >= 3  # <START> foo <END> need at least length 3.

        self.data_dir = data_dir
        self.split = split
        self.d_image_size = d_image_size
        self.transform = transform
        self.should_pad = should_pad
        self.pad_to_length = pad_to_length
        self.no_start_end = no_start_end
        self.return_decoded = return_decoded

        metadata = torch.load(os.path.join(self.data_dir, 'metadata.pth'))

        # labels
        self.img_id_to_class_id = metadata['img_id_to_class_id']
        self.class_id_to_class_name = metadata['class_id_to_class_name']
        self.class_name_to_class_id = metadata['class_name_to_class_id']

        # captions
        self.img_id_to_encoded_caps = metadata['img_id_to_encoded_caps']
        self.word_id_to_word = metadata['word_id_to_word']
        self.word_to_word_id = metadata['word_to_word_id']
        self.pad_token = self.word_to_word_id['<PAD>']
        self.start_token = self.word_to_word_id['<START>']
        self.end_token = self.word_to_word_id['<END>']
        self.unknown_token = self.word_to_word_id['<UNKNOWN>']

        self.d_vocab = metadata['num_words']
        self.num_captions_per_image = metadata['num_captions_per_image']

        # nlp = English()
        # self.tokenizer = nlp.Defaults.create_tokenizer(nlp) # Create a Tokenizer with the default settings for English including punctuation rules and exceptions
        from spacy.lang.en import English
        nlp = English()
        self.tokenizer = nlp.tokenizer  # Adapted to newer spacy version.

        # images
        if split == 'all':
            self.img_ids = metadata['img_ids']
            if d_image_size == 64:
                imgs_path = os.path.join(self.data_dir, 'imgs_64x64.pth')
            elif d_image_size == 128:
                imgs_path = os.path.join(self.data_dir, 'imgs_128x128.pth')
            else:
                # This is not included in the dataset, probably one needs to merge
                # the 'train_val' and 'test' splits.
                imgs_path = os.path.join(self.data_dir, 'imgs_256x256.pth')
        elif split == 'train_val':
            self.img_ids = metadata['train_val_img_ids']
            if d_image_size == 64:
                imgs_path = os.path.join(self.data_dir, 'imgs_train_val_64x64.pth')
            elif d_image_size == 128:
                imgs_path = os.path.join(self.data_dir, 'imgs_train_val_128x128.pth')
            else:
                imgs_path = os.path.join(self.data_dir, 'imgs_train_val_256x256.pth')
        else:
            self.img_ids = metadata['test_img_ids']
            if d_image_size == 64:
                imgs_path = os.path.join(self.data_dir, 'imgs_test_64x64.pth')
            elif d_image_size == 128:
                imgs_path = os.path.join(self.data_dir, 'imgs_test_128x128.pth')
            else:
                imgs_path = os.path.join(self.data_dir, 'imgs_test_256x256.pth')

        self.imgs = torch.load(imgs_path)
        assert self.imgs.size()[1:] == (3, d_image_size, d_image_size) and self.imgs.dtype == torch.uint8

    def encode_caption(self, cap):
        words = [token.text for token in self.tokenizer(cap)]
        return [self.word_to_word_id.get(word, self.unknown_token) for word in words]

    def decode_caption(self, cap):
        if isinstance(cap, torch.Tensor):
            cap = cap.tolist()
        return ' '.join([self.word_id_to_word[word_id] for word_id in cap])

    def pad_caption(self, cap):
        max_len = self.pad_to_length - 2  # 2 since we need a start token and an end token.
        cap = cap[:max_len]  # truncate to maximum length.
        cap = [self.start_token] + cap + [self.end_token]
        cap_len = len(cap)
        padding = [self.pad_token] * (self.pad_to_length - cap_len)
        cap += padding
        return torch.tensor(cap, dtype=torch.long), cap_len

    def pad_without_start_end(self, cap):
        cap = cap[:self.pad_to_length]  # truncate to maximum length.
        cap_len = len(cap)
        padding = [self.pad_token] * (self.pad_to_length - cap_len)
        cap += padding
        return torch.tensor(cap, dtype=torch.long), cap_len

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transform:
            img = transforms.functional.to_pil_image(img)
            img = self.transform(img)
        img_id = self.img_ids[idx]
        class_id = self.img_id_to_class_id[img_id]
        encoded_caps = self.img_id_to_encoded_caps[img_id]
        cap_idx = torch.randint(low=0, high=self.num_captions_per_image, size=(1,)).item()
        encoded_cap = encoded_caps[cap_idx]
        if self.return_decoded:
            decoded_cap = self.decode_caption(encoded_cap)
            cl_name = self.class_id_to_class_name[class_id]
            return img, class_id, cl_name, decoded_cap
        if self.should_pad:
            if self.no_start_end:
                encoded_cap, cap_len = self.pad_without_start_end(encoded_cap)
            else:
                encoded_cap, cap_len = self.pad_caption(encoded_cap)
            return img, class_id, encoded_cap, cap_len
        return img, class_id, encoded_cap


class CUB200Pairs(CUB_200_2011):

    def __init__(
            self, data_dir, split, d_image_size=256, transform=None, should_pad=False,
            pad_to_length=None, no_start_end=False, return_decoded=True, verbose=False,
            n_captions=2, text_processor=None, instruct=None):
        super().__init__(
            os.path.join(data_dir,'cub'), split, d_image_size, transform, 
            should_pad, pad_to_length, no_start_end, return_decoded)
        self.text_processor = text_processor
        self.instruct = instruct
        self.verbose = verbose
        self.unknown_str = self.decode_caption([self.unknown_token, ])
        self.n_captions = n_captions

    def __getitem__(self, idx):

        def _sample_cap(encoded_caps=[]):
            cap = self.unknown_str
            # print(cap)
            while self.unknown_str in cap:  # To avoid broken captions.
                cap_idx = random.randint(0, self.num_captions_per_image - 1)
                cap = self.decode_caption(encoded_caps[cap_idx])
            return cap

        # Sample image and true caption.

        img = self.imgs[idx]
        if self.transform:
            img = transforms.functional.to_pil_image(img)
            img = self.transform(img)
        img_id = self.img_ids[idx]
        class_id = self.img_id_to_class_id[img_id]
        encoded_caps = self.img_id_to_encoded_caps[img_id]
        cap = _sample_cap(encoded_caps)
        cls_name = self.class_id_to_class_name[class_id]

        # Sample a caption from an image from another class.
        class_id_alt = class_id
        while class_id_alt == class_id:
            idx_alt = random.randint(0, self.imgs.shape[0] - 1)
            img_id_alt = self.img_ids[idx_alt]
            class_id = self.img_id_to_class_id[img_id_alt]
        encoded_caps = self.img_id_to_encoded_caps[img_id_alt]
        cap_alt = _sample_cap(encoded_caps)
        caps = [cap, cap_alt]
        random.shuffle(caps)
        lab = int(caps[1] == cap)
        if self.verbose:
            print(f'true={cls_name}, {self.class_id_to_class_name[class_id]}')
        if self.instruct:
            caps = [self.text_processor(self.instruct.format(cap1=caps[0], cap2=caps[1]))]

        return img, *caps, lab
