import torch as torch
import torch.nn.functional as F
#from torch.utils.data import DataLoader

#from data import get_data
from eval.utils import parse_args, get_model_and_data


def eval(model, tokenizer, test_loader, args):
    """
    Task: given a text description and two images, decide which one better aligns
    with the text.
    """

    print(f'Evaluating {args.data} dataset.')

    if args.model_type == 'gen-lmmm':
       raise NotImplemented

    elif (args.model_type == 'embedding-model' or
          args.modelname in ['liqe-mix', 'liqe-koniq']):
        # Get encoders.
        _im_enc = model.encode_image
        _txt_enc = model.encode_text

        all, count = 0, 0
        with torch.no_grad():  # torch.cuda.amp.autocast()
            for e, (img, texts, lab) in enumerate(test_loader):
                if e == 0 and all == 0:
                    print(len(texts), lab)
                txts_emb = [_txt_enc(tokenizer(tks).to(args.device)) for tks in texts]
                img_emb = _im_enc(img.to(args.device))
                sims = [F.cosine_similarity(img_emb, txt_emb, dim=-1) for txt_emb in txts_emb]
                pred = torch.stack(sims, dim=-1).argmax(dim=-1)  
                count += (pred.long().to(lab.device) == lab).float().sum()
                all += img.shape[0]

                print(f'Clean accuracy: {count / all:.1%} ({count:.0f}/{all})')

                if all >= args.num_samples:
                    break

    elif args.model_type == 'score-model':
        raise NotImplemented


def main(args):
    
    tokenizer, model, dataloader = get_model_and_data(args)
    eval(model, tokenizer, dataloader, args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    args = parse_args()
    main(args)
