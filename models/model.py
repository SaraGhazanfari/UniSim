import os

import torch

PRETRAINED_MODELS = {
    'convnext_base_w-fare': {
        'basemodel': 'hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg',
        'ckptpath': 'convnext_base_w-fare-eps4.pt',
    },
    'convnext_base_w-tecoa': {
        'basemodel': 'hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg',
        'ckptpath': 'convnext_base_w-tecoa-eps4.pt',
    },
    'vit-b-16-fare': {
        'basemodel': 'hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K',
        'ckptpath': 'vit-b-16-fare-eps4.pt',
    },
    'vit-b-16-tecoa': {
        'basemodel': 'hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K',
        'ckptpath': 'vit-b-16-tecoa-eps4.pt',
    },
}


def resolve_ckpt(ckpt):

    if 'state_dict' in ckpt.keys():
        ckpt = ckpt['state_dict']
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    return ckpt


def get_model(args, device):
    if 'llava-next' in args.model_path or 'unisim' in args.model_path:
        from models.llava_next.mm_utils import get_model_name_from_path
        from models.llava_next.model.builder import load_pretrained_model
        model_path = os.path.expanduser(args.model_path)
        model_name = 'llava_qwen'
        try:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                    model_path,
                    args.model_base,
                    model_name,
                    device_map='auto',
                    use_cache=True,
                    cache_dir=args.cache_dir)
        except:
            if 'unisim-7b' in args.model_path:
                tokenizer, model, image_processor, context_len = load_pretrained_model(
                    'lmms-lab/llava-next-interleave-qwen-7b',
                    args.model_base,
                    model_name,
                    device_map='auto',
                    cache_dir=args.cache_dir,
                    use_cache=True)
                state_dict = torch.load(os.path.join(args.model_path, 'mm_projector.bin'))
                new_state_dict = dict()
                for key, value in state_dict.items():
                    new_state_dict[key.replace('model.mm_projector.', '')] = value
                model.model.mm_projector.load_state_dict(new_state_dict)
            else:
                raise Exception(f'{args.model_path} model is not supported!')
            
        return tokenizer, model, image_processor, context_len

    if 'llava' in args.model_path:
        from models.llava.mm_utils import get_model_name_from_path
        from models.llava.model.builder import load_pretrained_model
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                                               device=device, cache_dir=args.cache_dir)
        return tokenizer, model, image_processor, context_len

    if 'Mantis' in args.model_path:
        from transformers import AutoModelForVision2Seq, Idefics2Processor, AutoProcessor
        chat_template = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b").chat_template
        processor = AutoProcessor.from_pretrained(args.model_path, cache_dir=args.cache_dir)
        processor.chat_template = chat_template
        model = AutoModelForVision2Seq.from_pretrained(args.model_path, cache_dir=args.cache_dir, device_map="auto")
        return processor, model, processor, None

    if 'q-future' in args.model_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        processor = CLIPImageProcessor.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                     attn_implementation="eager",
                                                     trust_remote_code=True,
                                                     torch_dtype=torch.float16,
                                                     cache_dir=args.cache_dir,
                                                     device_map="auto")
        return None, model, None, None
    
    if 'Qwen' in args.model_path:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        # default: Load the model on the available device(s)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto", cache_dir=args.cache_dir
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", cache_dir=args.cache_dir)
        return processor, model, processor.image_processor, None

    if 'InternVL2_5' in args.model_path:
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            args.model_path,
            cache_dir='./',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir='./', trust_remote_code=True, use_fast=False)
        return tokenizer, model, None, None
    else:
        raise Exception(f'{args.model_path} model is not supported!')


def get_other_models(args):
    """Preliminary tool to load non-generative models."""
    if 'pac-s' in args.modelname:
        from models.pac_s.compute_metric import PACScore
        model = PACScore(device=args.device, modelname=args.modelname)
        return model.tokenizer, model, model.image_processor, None
    
    if args.ckptpath is not None or args.modelname.startswith('openai'):
        from train.open_clip_train.main import UniSim
        args.model = args.modelname
        model = UniSim(args, device=args.device)
        
        if args.ckptpath is not None:
            ckptpath = os.path.join(args.model_dir, args.ckptpath)
            print(f'Loading checkpoint from {ckptpath}.')
            ckpt = torch.load(ckptpath, map_location='cpu')
            ckpt = resolve_ckpt(ckpt)
            model.load_state_dict(ckpt, strict=True)
            
        model.to(args.device)
        model.eval()    
        return model.tokenizer_val, model, model.preprocess_val, None
    
    if args.modelname in [
        'hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg',
        'hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K',
        'hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft',
        'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        'hf-hub:timm/ViT-SO400M-14-SigLIP',
        'hf-hub:timm/ViT-B-16-SigLIP-512',
        'hf-hub:timm/ViT-B-16-SigLIP',
        'ViT-B-32',
        'hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
        'hf-hub:chs20/FARE4-convnext_base_w-laion2B-s13B-b82K-augreg',
        'hf-hub:chs20/FARE4-ViT-B-32-laion2B-s34B-b79K',
    ]:
        import open_clip
        model, _, image_preprocess = open_clip.create_model_and_transforms(
            args.modelname, pretrained=args.pretrained, cache_dir=args.model_dir,
        )
        
        if args.ckptpath is not None:
            ckptpath = os.path.join(args.model_dir, args.ckptpath)
            print(f'Loading checkpoint from {ckptpath}.')
            ckpt = torch.load(ckptpath, map_location='cpu')
            ckpt = resolve_ckpt(ckpt)
            model.load_state_dict(ckpt, strict=True)

        model.to(args.device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(args.modelname)

        return tokenizer, model, image_preprocess, None

    if args.modelname == 'hps':
        from models.hps.builder import build_model
        return build_model(args)

    if args.modelname in ['ImageReward-v1.0', 'blip']:
        from models.image_reward.builder import load_image_reward
        tokenizer_orig, model, image_processor, context_len = load_image_reward(
            name=args.modelname, device=args.device, download_root=args.cache_dir
        )

        # Tokenizer parameters copied from
        # https://github.com/THUDM/ImageReward/blob/f7f0d1aa3b54a5c6c6ac3109f0759aef76d2c13d/train/src/rank_pair_dataset.py#L86
        # to allow batched inference.
        def tokenizer(prompt):
            return tokenizer_orig(prompt, padding='max_length', truncation=True,
                                  max_length=35, return_tensors="pt"
                                  )

        model.eval()
        return tokenizer, model, image_processor, context_len

    if args.modelname in PRETRAINED_MODELS.keys():
        # Load fine-tuned vision encoder in existing CLIP models.
        import open_clip
        basemodel = PRETRAINED_MODELS[args.modelname]['basemodel']
        model, _, image_preprocess = open_clip.create_model_and_transforms(
            basemodel, pretrained=args.pretrained, cache_dir=args.model_dir,
        )
        ckpt = torch.load(
            os.path.join(args.model_dir, PRETRAINED_MODELS[args.modelname]['ckptpath']),
            map_location='cpu')
        model.visual.load_state_dict(ckpt, strict=True)
        model.to(args.device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(basemodel)
        if 'convnext_base_w' in args.modelname:
            # The original model was trained at 256px, but fine-tuned at 224 px.
            from torchvision import transforms
            # print(image_preprocess)
            image_preprocess.transforms[1] = transforms.CenterCrop(224)
        return tokenizer, model, image_preprocess, None
    
    if args.modelname in ['liqe-mix', 'liqe-koniq']:
        import models.liqe.liqe_arch
        from models.liqe.utils import liqe_prepr
        import clip
        model = models.liqe.liqe_arch.LIQE(
            cache_dir=args.cache_dir,
            batched_input=True,
            pretrained='liqe_koniq.pt' if args.modelname == 'liqe-koniq' else 'liqe_mix.pt')
        model.to(args.device)
        model.eval()
        tokenizer = lambda x: clip.tokenize(x, context_length=77, truncate=True)
        image_preprocess = liqe_prepr
        return tokenizer, model, image_preprocess, None

    if args.modelname.startswith('dreamsim:'):
        from dreamsim import dreamsim
        from torchvision import transforms

        def _convert_to_rgb(image):
            return image.convert('RGB')

        modelname = args.modelname.replace('dreamsim:', '')
        assert modelname in ('open_clip_vitb32', 'clip_vitb32', 'dino_vitb16', 'ensemble')
        model, preprocess = dreamsim(
            pretrained=True,
            cache_dir=args.cache_dir,
            dreamsim_type=modelname,
            device=args.device,  # Changing device after initialization has to
                                 # be done manually for each model part.
            )
        model.eval()
        print(preprocess)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            _convert_to_rgb,
            transforms.ToTensor()
        ])  # The preprocess loaded is a function.
        tokenizer = None
        
        if modelname in ['open_clip_vitb32']:
            # DreamSim model definition doesn't include the text encoder.
            # Adapted from https://github.com/ssundaram21/dreamsim/blob/main/dreamsim/feature_extraction/load_open_clip_as_dino.py.
            import open_clip
            _modelname = {
                'open_clip_vitb32': 'ViT-B-32'
            }[modelname]
            clip_all, _, _ = open_clip.create_model_and_transforms(
                _modelname, pretrained='laion400m_e31', cache_dir=args.cache_dir)
            clip_all.to(args.device)
            clip_all.eval()
            assert not hasattr(model, 'text_encoder')
            setattr(model, 'orig_clip', clip_all)
            tokenizer = open_clip.get_tokenizer(_modelname)

        return tokenizer, model, preprocess, None

    else:
        raise ValueError(f'Unknown model: {args.modelname}.')
