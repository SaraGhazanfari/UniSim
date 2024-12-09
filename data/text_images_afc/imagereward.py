import random

import torch

from torch.utils.data import Dataset


class ImageRewardPairs(Dataset):

    name = 'imagereward'

    def __init__(self, data_dir, img_prepr=None, text_prepr=None, size='1k', split=None,
                 verbose=False, instruct=None):
        from datasets import load_dataset
        self.dataset = load_dataset("THUDM/ImageRewardDB", size, cache_dir=data_dir,
                                    trust_remote_code=True)[split]
        self.img_preprocess = img_prepr
        self.text_preprocess = text_prepr
        if self.text_preprocess is None:
            self.text_preprocess = lambda x: x
        self.prompt_ids = self.dataset['prompt_id']
        self.unique_prompts_ids = list(set(self.prompt_ids))
        self.verbose = verbose
        self.instruct = instruct

    def collate_fn(self, examples):

        output_dict = {}
        imgs = [self.preprocess(ex['image']) for ex in examples]
        for k, v in examples[0].items():
            if k == 'image' and isinstance(v, torch.Tensor):
                output_dict[k] = torch.stack(imgs, dim=0)
                # continue
            output_dict[k] = [example[k] for example in examples]
        return output_dict

    def __getitem__(self, idx):

        # prompt_id = random.choice(self.unique_prompts_ids)
        prompt_id = self.unique_prompts_ids[idx]
        subset = [i for i, item in enumerate(self.prompt_ids) if item == prompt_id]
        # print(prompt_id, subset)
        subset = [subset[0], subset[-1]]  # These should be best and worst ranked images.
        # print(subset)
        random.shuffle(subset)
        # print(subset)
        data1 = self.dataset[subset[0]]
        data2 = self.dataset[subset[-1]]
        prompt = data1['prompt']  # Both images should have the same prompt
        if self.instruct:
            prompt = self.instruct.format(prompt=prompt)
        if self.text_preprocess:
            prompt = self.text_preprocess(prompt)
        assert data1['rank'] != data2['rank']
        lab = int(data1['rank'] > data2['rank'])  # 0 -> img1 is the better one (lower rank)
        if self.verbose:
            print('selected images', subset, f'rank 1={data1["rank"]}',
                  f'rank 2={data2["rank"]}', f'label={lab}')
        img1 = data1['image']
        img2 = data2['image']
        if self.img_preprocess is not None:
            img1 = self.img_preprocess(img1)
            img2 = self.img_preprocess(img2)
        return img1, img2, lab, prompt  # , prompt_id

    def __len__(self):
        return len(self.unique_prompts_ids)  # Number of unique prompts.
