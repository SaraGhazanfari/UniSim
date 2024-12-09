import os

import torch as torch
import torch.nn.functional as F

from eval.utils import parse_args, get_model_and_data


def get_text_embedding(args, dataloader, text_enc=None, tokenizer=None):
    _modelname = args.modelname.replace('/', '-')
    if args.pretrained is not None:
        _modelname += '_' + args.pretrained.replace('/', '_')
    text_emb_path = os.path.join(
        args.data_path, f'text_emb_{args.template}_{args.data}_{_modelname}.pth')

    if os.path.exists(text_emb_path):
        print(f'Loading embedding from {text_emb_path}.')
        return torch.load(text_emb_path, map_location='cpu')

    else:
        # Adapted from https://github.com/chs20/RobustVLM/blob/main/CLIP_eval/clip_robustbench.py
        if not args.template == 'ensemble':
            if args.template == 'std':
                template = 'This is a photo of a {}'
            else:
                raise ValueError(f'Unknown template: {args.template}.')
            print(f'template: {template}')
            text_emb = []
            for cls in dataloader.dataset.class_names:
                text = template.format(cls)
                tkns = tokenizer(text)
                with torch.no_grad():
                    cls_emb = text_enc(tkns.to(args.device))
                    # cls_emb = F.normalize(cls_emb, dim=-1)
                    text_emb.append(cls_emb.cpu())
                    # print(cls_emb.shape)
            text_emb = torch.cat(text_emb, dim=0)

        else:
            raise NotImplementedError()

        print(f'Saving embedding at {text_emb_path}.')
        torch.save(text_emb, text_emb_path)
        return text_emb


def eval(model, tokenizer, test_loader, args):
    """
    Task: zero-shot image classification.
    """

    print(f'Evaluating {args.data} dataset.')

    if args.model_type == 'gen-lmmm':
        raise NotImplementedError()


    elif args.model_type == 'perc-metric':
        raise NotImplementedError()

    elif args.model_type in ['embedding-model',  # 'score-model'
                             ]:
        # Get encoders.
        if args.modelname.startswith('dreamsim:'):
            _im_enc = model.embed
            _txt_enc = model.orig_clip.encode_text
        elif hasattr(model, 'encode_image'):
            _im_enc = model.encode_image
            _txt_enc = model.encode_text
        else:
            _im_enc = model.get_image_features
            _txt_enc = model.get_text_features

        text_emb = get_text_embedding(args, test_loader, _txt_enc, tokenizer)
        # text_emb = F.normalize(text_emb, dim=-1)
        text_emb = text_emb.T.unsqueeze(0).to(args.device)
        print(text_emb.shape)

        count, all = 0, 0
        with torch.no_grad():
            for e, (img, target) in enumerate(test_loader):

                if e == 0:
                    print(img.shape)
                img_emb = _im_enc(img.to(args.device))
                img_emb = img_emb.unsqueeze(-1)
                sim = F.cosine_similarity(img_emb, text_emb, dim=1)
                pred = sim.argmax(dim=-1)
                count += (pred.long().to(target.device) == target.squeeze()).float().sum()
                all += img.shape[0]

                print(f'Clean accuracy: {count / all:.1%} ({count:.0f}/{all})')

                if all >= args.num_samples:
                    break

    args.logger.log(f'dataset: {args.data}, clean accuracy: {count / all:.1%} ({count:.0f}/{all})')


def main(args):
    tokenizer, model, dataloader = get_model_and_data(args)
    eval(model, tokenizer, dataloader, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
