import argparse
import os

import torch
from torch.utils.data import DataLoader

from data import get_data
from models.model import get_model, get_other_models


def parse_args():
    parser = argparse.ArgumentParser()

    # Model. 
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top-p', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--model-path", type=str, default='other')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default='./')
    parser.add_argument("--attn_implementation", type=str, default='flash_attention_2')

    # For perceptual metrics models.
    parser.add_argument('--ckptpath', type=str, default=None)
    parser.add_argument('--shortname', type=str)
    parser.add_argument('--modelname', type=str)
    parser.add_argument('--mlp_head', type=str)
    parser.add_argument('--lora_weights', type=str)
    parser.add_argument('--pretrained', type=str)

    # LoRA
    parser.add_argument("--lora", action='store_true', default=False)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.2)

    # Data.
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data", type=str)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument('--model_dir', type=str, default='./')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--image-size', type=int, default=320)
    parser.add_argument('--template', type=str, default='std')

    # Others.
    parser.add_argument('--attack_name', type=str)
    parser.add_argument('--norm', type=str)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--log', action='store_true')

    args = parser.parse_args()

    # Adjust stuff.
    if not torch.cuda.is_available():
        args.device = "cpu"
    args.dataset = args.data
    args.n_ex = args.num_samples
    if args.model_dir is None:
        args.model_dir = args.cache_dir

    # Infer the type of model to be used.
    args.model_type = 'gen-lmmm'  # Generative multi-model LLMs.
    if args.shortname is not None:
        args.model_type = 'perc-metric'  # Implementation from the perceptual metrics paper.
        print(args.shortname)
    elif args.modelname is not None:
        args.model_type = 'embedding-model'  # CLIP-like models.
        if args.modelname in ['ImageReward-v1.0', 'blip', 'liqe-mix', 'liqe-koniq', 'pac-s', 'pac-s+']:
            args.model_type = 'score-model'  # Needs a different eval function.
        print(args.modelname)
    else:
        print(args.model_path)
    print(args.model_type)

    # # Fix seed.
    # args.shuffle_data = fix_seed(args)

    if args.log:
        get_logger(args)

    return args


def get_any_model(args):
    # passed via `--shortname` and multi-model LLMs via `--model-path` which does
    # not seem very stable.
    print(args.modelname)
    if args.model_type == 'gen-lmmm':
        tokenizer, model, image_processor, context_len = get_model(args, args.device)

    elif args.model_type == 'perc-metric':
        from perc_models import get_model_and_preprocess
        tokenizer = None
        context_len = None
        model, image_processor = get_model_and_preprocess(args)

    elif args.model_type in ['embedding-model', 'score-model']:
        context_len = None
        tokenizer, model, image_processor, _ = get_other_models(args)

    else:
        raise ValueError(f'Unknown model type: {args.model_type}')

    print('Model loaded.')

    return tokenizer, model, image_processor, context_len


def get_c2s_score(_img0, _img1, toks, input_ids, model, tokenizer):
    import numpy as np
    ids_ = [id_[1] for id_ in tokenizer(toks)["input_ids"]]
    with torch.inference_mode():
        output_logits = model(input_ids, images=[_img0, _img1])["logits"][:, -1]
    comparison = {}

    for tok, id_ in zip(toks, ids_):
        comparison[tok] = output_logits[0, id_].item()
    t = 100
    logits = np.array([comparison[toks[0]] / t, comparison[toks[1]] / t,
                       comparison[toks[2]] / t, comparison[toks[3]] / t,
                       comparison[toks[4]] / t])
    probability = np.exp(logits) / sum(np.exp(logits))
    preference = np.inner(probability, np.array([0, 0.25, 0.5, 0.75, 1.]))
    return preference


def fix_seed(args):
    if args.seed is None:
        print('No seed given, the dataset will not be shuffled.')
        return False

    else:
        import random
        import numpy as np
        print(f'Fixing seed={args.seed}.')
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)  # For multi-GPU
        return True


def get_model_and_data(args):
    tokenizer, model, image_processor, context_len = get_any_model(args)

    # Fix seed.
    args.shuffle_data = fix_seed(args)

    if args.data not in ['roxford5k', 'rparis6k']:
        test_dataset = get_data(
            args, tokenizer=tokenizer, image_processor=image_processor, split=args.split)
        dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=args.shuffle_data)
        print(f'Data loaded ({len(test_dataset)} data points).')

        return tokenizer, model, dataloader

    else:
        train_dataset = get_data(
            args, tokenizer=tokenizer, image_processor=image_processor, split='train')
        query_dataset = get_data(
            args, tokenizer=tokenizer, image_processor=image_processor, split='query')
        dataloaders = [
            DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False),
            DataLoader(query_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False)]
        print(f'Data loaded ({len(train_dataset)} + {len(query_dataset)} data points).')

        return tokenizer, model, dataloaders


class Logger():
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()


def get_logger(args):
    if args.logdir is None:
        args.logger = Logger(None)

    else:
        os.makedirs(args.logdir, exist_ok=True)
        logpath = 'log_'
        if args.ckptpath is not None:
            logpath += args.ckptpath.replace('/', '_')
        else:
            logpath += args.modelname.replace('/', '_')
        if args.pretrained is not None:
            logpath += '_' + args.pretrained.replace('/', '_')
        logpath += '.txt'
        logpath = os.path.join(args.logdir, logpath)

        args.logger = Logger(logpath)
