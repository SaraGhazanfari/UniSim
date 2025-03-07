import copy
import os

from data.utils import build_transform, dynamic_preprocess
import torch
import torchvision

from torchvision import datasets, transforms

from data.iqa.kadid10k import KADID10kPairs
from data.iqa.koniq10k import KonIQ10kPairs
from data.iqa.pipal import PiPal
from data.iqa.pie_app import PieAppIQADataset
from data.prompt_constants import PROMPT_DICT, DATASET_TYPE, _get_model_type
from data.iqa.sice import SICEPairs
from data.retrieval.roxford_rparis import OxfordParisDataset
from data.text_2afc.coco import CocoTriplets
from data.text_2afc.cub import CUB200Pairs
from data.text_2afc.foil import Foil
from data.text_2afc.amber import AMBERDataset
from data.text_2afc.hq_edit import HQEdit
from data.text_2afc.polaris import Polaris
from data.text_images_afc.agiqa import AGIQA3KPairs
from data.text_images_afc.birds_to_words import BirdsToWords
from data.text_images_afc.hpdv2 import HPDv2Pairs
from data.text_images_afc.imagereward import ImageRewardPairs
from data.text_images_afc.magic_brush import MagicBrush
from data.text_images_afc.spot_the_diff import SpotTheDiff
from data.three_afc.cifar import CIFAR100Coarse, CIFAR100CoarseTriplets
from data.three_afc.imagenet import ImageNetTriplets, RestrictedImageNetTriplets, BalRestrictedImageNetTriplets, \
    HardImageNetTriplets, ImageNet
from data.three_afc.things import THINGSDataset
from data.two_afc.bapps import BAPPSDataset
from data.two_afc.night import NightDataset
from data.two_afc.pie_app import PieApp2AFCDataset



def preprocess_prompt(args, tokenizer):
    question = PROMPT_DICT[_get_model_type(args.model_path)][DATASET_TYPE[args.data]] # question is dependent on the model and dataset chosen
    # print(f'Here is the instruction: {question}')
    return get_tokenizer(args, tokenizer)(question)


def get_processor(args, image_processor):
    # if isinstance(image_processor, Idefics2Processor):
    
    if 'Mantis' in args.model_path:
        resize = torchvision.transforms.Resize((args.image_size, args.image_size))
        _prep = lambda x: image_processor(images=resize(x), return_tensors='pt')
        return _prep
    
    if 'q-future' in args.model_path:
        return None
    
    if 'llava-next' in args.model_path or 'unisim' in args.model_path:
        from models.llava_next.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
        _prep = lambda x: image_processor.preprocess(images=x, return_tensors='pt')
        return _prep
    if 'InternVL2_5' in args.model_path:
        def load_image(image, input_size=448, max_num=12):

            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values
        return load_image
    
    if 'Qwen' in args.model_path:
        return None
    
    if (isinstance(image_processor, transforms.transforms.Compose) or
          args.modelname in ['liqe-mix',
                             'liqe-koniq',
                             'wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M']):
        return image_processor


    return lambda x: image_processor(images=x, return_tensors='pt')


def get_tokenizer(args, tokenizer):

    
    if args.model_type == 'gen-lmmm':
        if 'Qwen' in args.model_path or 'InternVL2_5' in args.model_path or 'q-future' in args.model_path:
            return None
        
        def _prep(x):

            if 'llava-next' in args.model_path or 'unisim' in args.model_path:
                from models.llava_next.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
                from models.llava_next.conversation import conv_templates
                from models.llava_next.mm_utils import tokenizer_image_token
                conv_template = "qwen_1_5"
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], x)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                return input_ids

            if 'llava' in args.model_path:
                from models.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
                from models.llava.conversation import conv_templates
                from models.llava.mm_utils import tokenizer_image_token
                prompt_question = DEFAULT_IMAGE_TOKEN + '\n' + x
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                return input_ids

            if 'Mantis' in args.model_path:
                from models.llava_next.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
                messages = [{
                    "role": "user",
                    "content": [{"type": "image"},
                                {"type": "image"}, ]}]
                if DATASET_TYPE[args.data] in ['text_2afc']:
                    messages[0]['content'] = [{"type": "image"}]
                elif DATASET_TYPE[args.data] in ['2afc', '3afc','text_images_many']:
                    messages[0]['content'].append({"type": "image"})
                messages[0]['content'].append({"type": "text", "text": x})
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                input_ids = tokenizer(text=prompt, return_tensors="pt")
                return input_ids

            if 'q-future' in args.model_path:
                from models.llava.constants import IMAGE_TOKEN_INDEX
                from models.llava.mm_utils import tokenizer_image_token
                from models.llava.conversation import conv_mplug_owl2
                conv_mode = "mplug_owl2"
                conv = conv_mplug_owl2.copy()
                conv.append_message(conv.roles[0], x)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                return input_ids
            
            return x

        return _prep

    elif args.model_type == 'perc-metric':
        return None

    elif args.model_type in ['embedding-model', 'score-model']:
        # return lambda x: {'input_ids': tokenizer(x,)}
        return None  # Tokenization happens in the eval function.


def get_data(args, tokenizer, image_processor, split='test'):
    _prep = get_processor(args, image_processor)
    print(_prep)
    _tokenizer = get_tokenizer(args, tokenizer)
    _instruct = PROMPT_DICT.get(_get_model_type(args.model_path), None)
    if _instruct is not None:
        _instruct = _instruct[DATASET_TYPE[args.data]]
    # Load dataset.
    if args.data == 'night':
        test_dataset = NightDataset(root_dir=args.data_path, split=split, image_processor=_prep)
        return test_dataset

    if args.data == 'bapps':
        test_dataset = BAPPSDataset(root_dir=args.data_path, split=f'val', image_processor=_prep, ratio=0.15)
        return test_dataset

    if args.data == 'things':
        ds = THINGSDataset(data_dir=args.data_path,
                           image_processor=_prep,
                           n_triplets_max=args.num_samples)
        return ds

    if args.data == 'cifar100':
        ds = datasets.CIFAR100(
            root=args.data_path,
            train=split == 'train',
            transform=_prep,  # transforms.Compose([transforms.ToTensor()]),
            download=True)
        return ds

    if args.data == 'cifar100coarse':
        ds = CIFAR100Coarse(
            root=args.data_path,
            train=split == 'train',
            transform=_prep,
            download=True)
        return ds

    if args.data == 'cifar100coarse-triplets':
        ds = CIFAR100CoarseTriplets(
            root=args.data_path,
            train=split == 'train',
            transform=_prep,
            download=True)
        return ds

    if args.data == 'imagenet':
        ds = ImageNet(
            data_dir=os.path.join(args.data_path, 'imagenet/val'), transforms_fn=_prep)
        return ds

    if args.data == 'imagenet-triplets':
        ds = ImageNetTriplets(
            data_dir=os.path.join(args.data_path, 'imagenet/val'), transforms_fn=_prep)
        return ds

    if args.data == 'r-imagenet-triplets':
        ds = RestrictedImageNetTriplets(
            data_dir=os.path.join(args.data_path, 'imagenet/val'), transforms_fn=_prep)
        return ds

    if args.data == 'bal-r-imagenet-triplets':
        ds = BalRestrictedImageNetTriplets(
            data_dir=os.path.join(args.data_path, 'imagenet/val'), transforms_fn=_prep)
        return ds

    if args.data == 'h-imagenet-triplets':
        ds = HardImageNetTriplets(
            data_dir=args.data_path, split='val', transforms_fn=_prep)
        return ds

    if args.data == 'imagereward1k-pairs':
        ds = ImageRewardPairs(
            data_dir=args.data_path, img_prepr=_prep, text_prepr=_tokenizer, split=split,
            instruct=_instruct)
        return ds

    if args.data in ['kadid-pairs', 'easy-kadid-pairs', 'kadid-corr-pairs']:
        _data_path = args.data_path
        _sampling_mode = {
            'kadid-pairs': 'sev-noref',
            'easy-kadid-pairs': 'easy-sev-noref',
            'kadid-corr-pairs': 'corr-noref',
        }[args.data]
        ds = KADID10kPairs(
            root_dir=_data_path, transform_fn=_prep, verbose=False, sampling_mode=_sampling_mode, split='test')
        return ds

    if 'hpdv2' in args.data: #in ['hpdv2-pairs', 'hpdv2-easy-pairs', 'hpdv2-mindist-pairs', 'hpdv2-3a', 'hpdv2-9a']:

       
        if args.data in ['hpdv2-pairs', 'hpdv2-easy-pairs', 'hpdv2-mindist-pairs']:
            _n_imgs = 2
            _sampling_mode = {
            'hpdv2-pairs': 'rand-all',
            'hpdv2-easy-pairs': 'easy',
            'hpdv2-mindist-pairs': 'min-dist'
        }[args.data]
        else:
            _n_imgs = int(args.data.split('-')[1].replace('a',''))
            _sampling_mode = 'rand-all'

        ds = HPDv2Pairs(
            data_dir=args.data_path, img_prepr=_prep, text_prepr=_tokenizer, split=split,
            instruct=_instruct, sampling_mode=_sampling_mode, n_imgs=_n_imgs,
            min_dist=3 if _sampling_mode == 'min-dist' else 0)
        return ds

    if args.data in ['cub-pairs']:
        _data_path = args.data_path
        ds = CUB200Pairs(_data_path, split=split, transform=_prep, text_processor=_tokenizer, instruct=_instruct)
        return ds

    if args.data in [
        'coco-triplets', 'coco-gen-triplets', 'h-coco-triplets', 'h-coco-gen-triplets']:
        _scores = {
            'coco-triplets': 'coco-expertscores',
            'coco-gen-triplets': 'coco-genericscores',
            'h-coco-triplets': 'coco-expertscores',
            'h-coco-gen-triplets': 'coco-genericscores',
        }[args.data]
        ds = CocoTriplets(
            data_dir=args.data_path,
            score_type=_scores,
            transform_fn=_prep,
            min_score=2 if args.data.startswith('h-') else 3,
            sampling_mode='rand' if args.data.startswith('h-') else 'all-zeros',
            text_processor=_tokenizer, instruct=_instruct
        )
        return ds

    if args.data in [
        'koniq-pairs', 'koniq-colorfulness-pairs', 'koniq-brightness-pairs',
        'koniq-contrast-pairs', 'koniq-sharpness-pairs']:
        _attr = {
            'koniq-pairs': 'quality',
            'koniq-colorfulness-pairs': 'colorfulness',
            'koniq-brightness-pairs': 'brightness',
            'koniq-contrast-pairs': 'contrast',
            'koniq-sharpness-pairs': 'sharpness',
        }[args.data]
        args.attribute = _attr
        ds = KonIQ10kPairs(
            split=split,
            attribute=_attr,
            data_dir=args.data_path,
            transform_fn=_prep,
            sampling_mode='rand-all',
        )
        return ds
    
    if args.data == 'birds_to_words':
        ds = BirdsToWords(
            data_dir=args.data_path, img_prepr=_prep, text_prepr=_tokenizer,
            split='test', test_size=1397, instruct=_instruct)
        return ds
    
    if args.data == 'hq_edit_text_2afc':
        ds = HQEdit(
            data_dir=args.data_path, img_prepr=_prep, text_prepr=_tokenizer,
            split=split, task_type='text_2afc', instruct=_instruct)
        return ds
    
    if args.data == 'hq_edit_text_images':
        ds = HQEdit(
            data_dir=args.data_path, img_prepr=_prep, text_prepr=_tokenizer,
            split=split, task_type='text_imags_fc', instruct=_instruct)
        return ds
    
    if args.data == 'magic_brush':
        ds = MagicBrush(
            data_dir=args.data_path, img_prepr=_prep, text_prepr=_tokenizer,
            split=split, instruct=_instruct)
        return ds
    
    if args.data == 'polaris':
        ds = Polaris(
            data_dir=args.data_path, img_prepr=_prep, text_prepr=_tokenizer,
            split=split, instruct=_instruct)
        return ds
    
    if args.data == 'spot_the_diff':
        ds = SpotTheDiff(
            data_dir=args.data_path, img_prepr=_prep, text_prepr=_tokenizer,
            split='test', instruct=_instruct)
        return ds
    
    if args.data == 'foil':
        ds = Foil(
            data_dir=args.data_path, img_prepr=_prep, text_prepr=_tokenizer,
            split=split, instruct=_instruct)
        return ds

    if args.data in ['agiqa3k-align-pairs', 'agiqa3k-qual-pairs', 'agiqa3k-align-3a']:
        _metric = {
            'agiqa3k-align-pairs': 'mos_align',
            'agiqa3k-qual-pairs': 'mos_quality',
            'agiqa3k-align-3a': 'mos_align',
        }[args.data]
        _n_imgs = {
            'agiqa3k-align-pairs': 2,
            'agiqa3k-qual-pairs': 2,
            'agiqa3k-align-3a': 3,
        }[args.data]
        ds = AGIQA3KPairs(
            data_dir=args.data_path,
            transform=_prep,
            metric=_metric,
            min_mos_quality=1. if _metric == 'mos_align' else 0,
            n_imgs=_n_imgs,
            text_preprocess=_tokenizer,
            instruct=_instruct)
        return ds

    if args.data in ['sice-pairs', 'sice-pairs-h']:
        args.attribute = 'brightness'
        ds = SICEPairs(
            data_dir=args.data_path,
            transform_fn=_prep,
            sampling_mode='rand-h' if args.data == 'sice-pairs-h' else 'rand-all',
            split=split,
        )
        return ds
    
    if args.data == 'pie-app-2afc':
        ds = PieApp2AFCDataset(
            data_dir=args.data_path, image_processor=_prep, split=split,)
        return ds
    
    if args.data == 'pie-app-iqa':
        ds = PieAppIQADataset(
            data_dir=args.data_path, image_processor=_prep, split=split,)
        return ds
    
    if args.data == 'pipal':
        ds = PiPal(
            data_dir=args.data_path, image_processor=_prep, split=split,)
        return ds
    
    elif args.data in ['roxford5k', 'rparis6k']:
        ds = OxfordParisDataset(
            dir_main=args.data_path, dataset=args.data, split=split, transform=_prep)
        return ds
    
    elif 'amber' in args.data:
        
        _n_texts = int(args.data.split('-')[1].replace('a',''))

        ds = AMBERDataset(
            data_dir=args.data_path, img_prepr=_prep, text_prepr=_tokenizer, split=split,
            instruct=_instruct, n_text=_n_texts)
        return ds
    
    elif 'flickr' in args.data:
        
        _n_texts = int(args.data.split('-')[1].replace('a',''))

        ds = Flickr_8k(
            data_dir=args.data_path, img_prepr=_prep, text_prepr=_tokenizer, split=split,
            instruct=_instruct, n_texts=_n_texts)
        return ds

    else:
        raise Exception('This data is not supported')
