import os

import huggingface_hub
import open_clip
import torch
from open_clip import get_tokenizer

from models.llava_next.model.multimodal_encoder.dev_eva_clip.eva_clip import create_model_and_transforms

hps_version_map = {
    "v2.0": "HPS_v2_compressed.pt",
    "v2.1": "HPS_v2.1_compressed.pt",
}


def build_model(args, hps_version: str = "v2.0"):
    # model, preprocess_train, preprocess_val = create_model_and_transforms(
    #     'ViT-H-14',
    #     'laion2B-s32B-b79K',
    #     precision='amp',
    #     device=args.device,
    #     jit=False,
    #     force_quick_gelu=False,
    #     force_custom_text=False,
    #     force_patch_dropout=False,
    #     force_image_size=None,
    #     pretrained_image=False,
    #     image_mean=None,
    #     image_std=None,
    #     light_augmentation=True,
    #     aug_cfg={},
    #     output_dict=True,
    #     with_score_predictor=False,
    #     with_region_predictor=False,
    #     cache_dir=args.cache_dir
    # )
    # tokenizer = get_tokenizer('ViT-H-14')
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K', cache_dir=args.cache_dir)
    tokenizer = open_clip.get_tokenizer(
        'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    # check if the checkpoint exists
    #if not os.path.exists(args.cache_dir):
    #    os.makedirs(args.cache_dir)
    cp = huggingface_hub.hf_hub_download(
        "xswu/HPSv2", hps_version_map[hps_version], cache_dir=args.cache_dir)

    checkpoint = torch.load(cp, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    model.to(args.device)
    model.eval()

    return tokenizer, model, preprocess_val, None

