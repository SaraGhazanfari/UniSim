"""
Adapted merging parts from https://github.com/zwx8981/LIQE
and https://github.com/chaofengc/IQA-PyTorch.
"""


import torch
import torch.nn as nn
import os

import clip
import torch.nn.functional as F
from itertools import product

from .clip_model import load

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']
dists_map = ['jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color', 'contrast', 'overexposure',
            'underexposure', 'spatial', 'quantization', 'other']

default_model_urls = {'koniq': 'https://github.com/zwx8981/IQA-PyTorch/releases/download/Weights/liqe_koniq.pt',
                      'mix': 'https://github.com/zwx8981/IQA-PyTorch/releases/download/Weights/liqe_mix.pt'}


class LIQE(nn.Module):
    def __init__(self,
                 model_type='liqe',
                 backbone='ViT-B/32',
                 step=32,
                 num_patch=15,
                 pretrained='liqe_mix.pt',
                 mtl=False,
                 cache_dir=None,
                 normalize_input=True,
                 batched_input=False,
                 ) -> None:
        super().__init__()
        assert backbone == 'ViT-B/32', 'Only support ViT-B/32 now'
        self.backbone = backbone
        # self.clip_model, self.preprocess = clip.load(
        #     backbone, device=device, jit=False, download_root=cache_dir)
        self.clip_model = load(self.backbone, 'cpu', download_root=cache_dir)
        self.model_type = model_type

        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)

        self.clip_model.logit_scale.requires_grad = False

        self.step = step
        self.num_patch = num_patch

        self.normalize_input = normalize_input
        self.batched_input = batched_input

        ckpt = os.path.join(cache_dir, pretrained)
        checkpoint = torch.load(ckpt, map_location='cpu')['params']
        self.load_state_dict(checkpoint)

        if pretrained == 'liqe_mix.pt':
            self.mtl = True
            text_feat_cache_path = os.path.join(cache_dir, "liqe_text_feat_mix.pt")
        else:
            self.mtl = mtl
            text_feat_cache_path = os.path.join(cache_dir, "liqe_text_feat.pt")

        if os.path.exists(text_feat_cache_path):
            self.text_features = torch.load(text_feat_cache_path, map_location='cpu', weights_only=False)
        else:
            print(f'Generating text features for LIQE model, will be cached at {text_feat_cache_path}.')
            if self.mtl:
                self.joint_texts = torch.cat(
                    [clip.tokenize(f"a photo of a {c} with {d} artifacts, which is of {q} quality") for q, c, d
                     in product(qualitys, scenes, dists_map)])
            else:
                self.joint_texts = torch.cat([clip.tokenize(f"a photo with {c} quality") for c in qualitys])

            self.text_features = self.get_text_features(self.joint_texts)
            torch.save(self.text_features.to('cpu'), text_feat_cache_path)
    
    def get_text_features(self, x):
        text_features = self.clip_model.encode_text(self.joint_texts.to(x.device))
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def forward_single(self, x):
        
        bs = x.size(0)
        h = x.size(2)
        w = x.size(3)

        assert (h >= 224) & (w >= 224), 'Short side is less than 224, try upsampling the original image'
        # preprocess image
        if self.normalize_input and not self.batched_input:
            x = (x - self.default_mean.to(x)) / self.default_std.to(x)

        x = x.unfold(2, 224, self.step).unfold(3, 224, self.step).permute(0, 2, 3, 1, 4, 5).reshape(bs, -1, 3, 224, 224)

        if x.size(1) < self.num_patch:
            num_patch = x.size(1)
            self.num_patch = num_patch
        else:
            num_patch = self.num_patch

        if self.training:
            sel = torch.randint(low=0, high=x.size(0), size=(num_patch, ))
        else:
            sel_step = max(1, x.size(1) // self.num_patch)
            sel = torch.zeros(num_patch)
            for i in range(num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        x = x[:, sel, ...]
        x = x.reshape(bs, num_patch, x.shape[2], x.shape[3], x.shape[4])

        text_features = self.text_features.to(x)
        x = x.view(bs * x.size(1), x.size(2), x.size(3), x.size(4))
        image_features = self.clip_model.encode_image(x, pos_embedding=True)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        logits_per_image = logits_per_image.view(bs, self.num_patch, -1)
        logits_per_image = logits_per_image.mean(1)
        logits_per_image = F.softmax(logits_per_image, dim=1)

        if self.mtl:
            logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists_map))
            logits_quality = logits_per_image.sum(3).sum(2)
        else:
            logits_per_image = logits_per_image.view(-1, len(qualitys))
            logits_quality = logits_per_image

        quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                             4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
        return quality
    
    def forward_batch(self, x):

        batch_size = x.size(0)
        num_patch = x.size(1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))

        logits_per_image = self.forward_single(x)
        logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
        logits_per_image = logits_per_image.mean(1)

        return logits_per_image
    
    def forward(self, x):

        if self.batched_input:
            return self.forward_batch(x).squeeze()
        return self.forward_single(x)
    
    # A bit hacky, as they don't natively support this.
    def encode_image(self, x):
        bs = x.shape[0]
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        fts = self.clip_model.encode_image(x, pos_embedding=True)
        return fts.view(bs, self.num_patch, -1).mean(1)
    
    def encode_text(self, x):
        return self.clip_model.encode_text(x)