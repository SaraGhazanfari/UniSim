'''
@File       :   ImageReward.py
@Time       :   2023/01/28 19:53:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   ImageReward Reward model.
* Based on CLIP code base and improved-aesthetic-predictor code base
* https://github.com/openai/CLIP
* https://github.com/christophschuhmann/improved-aesthetic-predictor
'''

import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

from models.blip import BLIP_Pretrain


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1)
        )

        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (self.input_size + 1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

    def forward(self, input):
        return self.layers(input)


class ImageReward(nn.Module):
    def __init__(self, med_config, device='cpu'):
        super().__init__()
        self.device = device

        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)
        self.preprocess = _transform(224)
        self.mlp = MLP(768)

        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

    def score(self, prompt, image):
        image_embeds = self.blip.visual_encoder(image)
        input_ids = prompt['input_ids'] if 'input_ids' in prompt else prompt.input_ids
        attn_mask = prompt['attention_mask'] if 'attention_mask' in prompt else prompt.attention_mask
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(input_ids,
                                             attention_mask=attn_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True,
                                             )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std

        return rewards.squeeze(1)  # .detach().cpu().numpy().item()

    def inference_rank(self, prompt, images):
        image_embeds = self.blip.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(prompt.input_ids,
                                             attention_mask=prompt.attention_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True,
                                             )
        txt_set = text_output.last_hidden_state[:, 0, :]

        txt_features = torch.cat(txt_set, 0).float()  # [image_num, feature_dim]
        rewards = self.mlp(txt_features)  # [image_num, 1]
        rewards = (rewards - self.mean) / self.std
        rewards = torch.squeeze(rewards)
        _, rank = torch.sort(rewards, dim=0, descending=True)
        _, indices = torch.sort(rank, dim=0)
        indices = indices + 1

        return indices, rewards
