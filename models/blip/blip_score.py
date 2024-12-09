import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

from models.blip.blip_pretrain import BLIP_Pretrain


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


class BLIPScore(nn.Module):
    def __init__(self, med_config, device='cpu'):
        super().__init__()
        self.device = device

        self.preprocess = _transform(224)
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)

    def score(self, prompt, image):
        input_ids = prompt['input_ids'] if 'input_ids' in prompt else prompt.input_ids
        attn_mask = prompt['attention_mask'] if 'attention_mask' in prompt else prompt.attention_mask
        text_output = self.blip.text_encoder(input_ids, attention_mask=attn_mask, mode='text')
        txt_feature = F.normalize(self.blip.text_proj(text_output.last_hidden_state[:, 0, :]))

        # image encode
        image_embeds = self.blip.visual_encoder(image)
        image_features = F.normalize(self.blip.vision_proj(image_embeds[:, 0, :]), dim=-1)
        # score
        rewards = torch.sum(torch.mul(txt_feature, image_features), dim=1, keepdim=True)

        return rewards.squeeze(1)
