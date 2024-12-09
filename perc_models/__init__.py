import torch
from torchvision import transforms
import os

from autoattack.other_utils import Logger

from perc_models.utils_perceptual_models import get_model_and_transforms, \
    PRETRAINED_MODELS, ClipVisionModel, ProjModel, MLP
from perc_models.utils_perceptual_eval import resolve_args


def get_model_and_preprocess(args, **kwargs):
    """Load model and pre-processing to use."""

    resolve_args(args)
    logger = Logger(args.log_path)
    logger.log(args.log_path)
    args.logger = logger

    model, preprocess = get_model_and_transforms(
        modelname=args.modelname,
        ckptpath=args.ckptpath,
        pretrained=args.pretrained,
        source=args.source,
        device=args.device,
        mlp_head=args.mlp_head,
        lora_weights=args.lora_weights,
        logger=args.logger,
        model_dir=args.model_dir,
        )
    model.eval()

    if args.mlp_head is not None:
        mlp_path, fts = PRETRAINED_MODELS[args.shortname]['mlp_info']
        mlp_path = os.path.join(args.model_dir, mlp_path)
        logger.log(f'Loading MLP head from {mlp_path}.')
        #mlp = MLP(*fts)
        mlp = MLP(*fts)
        if mlp_path is not None:
            ckpt_mlp = torch.load(mlp_path, map_location='cpu')
            mlp.load_state_dict(
                {k.replace('perceptual_model.mlp.', ''): v for k, v in ckpt_mlp['state_dict'].items()})
        mlp.eval()
        mlp.to(args.device)

    if args.source in ['openclip', 'clip']:
        # Move normalization to FP.
        norm_layer = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711))
        preprocess.transforms = preprocess.transforms[:-1]
        normalize_fn = norm_layer

        if args.lora_weights is None:
            vis_enc = ClipVisionModel(model.visual, None, normalize_fn)
        else:
            import perc_models.utils_lora
            #lora_path = utils_perceptual_models.LORA_WEIGHTS_DICT[args.lora_weights]
            lora_path = PRETRAINED_MODELS[args.shortname]['lora_path']
            lora_path = os.path.join(args.model_dir, lora_path)
            logger.log(f'Loading LoRA weights from {lora_path}.')
            # Following needed for using models fine-tuned as in DreamSim.
            lora_model, lora_proj = perc_models.utils_lora.load_lora_models(
                model.visual,
                args.arch.replace('+lora', '').replace('+head', ''),
                lora_path)
            if lora_proj is not None:
                lora_model = ProjModel(lora_model, lora_proj)
            vis_enc = ClipVisionModel(lora_model, None, normalize_fn)
        vis_enc.eval()
        vis_enc.to(args.device)

        if args.mlp_head is None:
            fp = lambda x: vis_enc(x, output_normalize=False)
        else:
            def fp(x):
                out = vis_enc(x, output_normalize=False)
                return mlp(out)

    return fp, preprocess