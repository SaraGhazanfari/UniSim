'''
@File       :   utils.py
@Time       :   2023/04/05 19:18:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
* Based on CLIP code base
* https://github.com/openai/CLIP
* Checkpoint of CLIP/BLIP/Aesthetic are from:
* https://github.com/openai/CLIP
* https://github.com/salesforce/BLIP
* https://github.com/christophschuhmann/improved-aesthetic-predictor
'''

import os, copy
from typing import Union, List

import torch
from huggingface_hub import hf_hub_download

from models.image_reward.model import ImageReward
from models.blip.blip_score import BLIPScore

_MODELS = {
    "ImageReward-v1.0": "https://huggingface.co/THUDM/ImageReward/blob/main/ImageReward.pt",
    "blip": 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth'
}


def available_models() -> List[str]:
    """Returns the names of available ImageReward models"""
    return list(_MODELS.keys())


def ImageReward_download(url: str, root: str, name='ImageReward-v1.0'):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)
    if os.path.exists(download_target):
        return download_target
    if name == 'blip':
        import wget
        wget.download(url=url, out=root)
    else:
        hf_hub_download(repo_id="THUDM/ImageReward", filename=filename, local_dir=root)
    return download_target


def load_image_reward(name: str = "ImageReward-v1.0",
                      device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                      download_root: str = None,
                      med_config: str = None):
    """Load a ImageReward model

    Parameters
    ----------
    name : str
        A model name listed by `ImageReward.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    download_root: str
        path to download the model files; by default, it uses "~/.cache/ImageReward"

    Returns
    -------
    model : torch.nn.Module
        The ImageReward model
    """
    if name in _MODELS:
        model_path = ImageReward_download(_MODELS[name], download_root, name=name)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    print('load checkpoint from %s' % model_path)
    state_dict = torch.load(model_path, map_location='cpu')
    # med_config
    if med_config is None:
        med_config = ImageReward_download("https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json",
                                          download_root)
    if name == 'blip':
        state_dict = state_dict['model']
        model = BLIPScore(device=device, med_config=med_config).to(device)
        msg = model.blip.load_state_dict(state_dict, strict=False)

    else:
        model = ImageReward(device=device, med_config=med_config).to(device)
        msg = model.load_state_dict(state_dict, strict=False)
    print(f"checkpoint loaded {msg}")
    model.eval()

    return model.blip.tokenizer, model, model.preprocess, None
