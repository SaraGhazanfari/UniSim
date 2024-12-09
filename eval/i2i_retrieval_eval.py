import torch as torch
import torch.nn.functional as F

from eval.utils import parse_args, get_model_and_data



def eval(model, tokenizer, test_loader, args):
    """
    Task: given are reference image and two alternatives, decide which is more similar
    to the reference.
    """

    print(f'Evaluating {args.data} dataset.')

    if args.model_type == 'gen-lmmm':
        
        raise NotImplementedError


    elif args.model_type == 'perc-metric':
        raise NotImplementedError

    elif args.model_type in ['embedding-model', 'score-model']:
        # Get image encoder.
        if args.model_type == 'embedding-model':
            if args.modelname.startswith('dreamsim:'):
                _enc = model.embed
            else:
                _enc = model.encode_image
        elif args.model_type == 'score-model':
            if args.modelname in ['liqe-mix', 'liqe-koniq', 'pac-s', 'pac-s+']:
                _enc = model.encode_image
            else:
                _enc = lambda x: model.blip.visual_encoder(x)[:, 0]

    train_loader, query_loader = test_loader

    train_emb, query_emb = [], []
    with torch.no_grad():
        # Get training features.
        for e, (img, _) in enumerate(train_loader):
            if e == 0:
                print(img.shape, img.min(), img.max())
            emb = _enc(img.to(args.device))
            train_emb.append(emb.cpu())
        train_emb = torch.cat(train_emb, dim=0)

        # Get query features.
        for e, (img, _) in enumerate(query_loader):
            if e == 0:
                print(img.shape)
            emb = _enc(img.to(args.device))
            query_emb.append(emb.cpu())
        query_emb = torch.cat(query_emb, dim=0)

        train_emb = F.normalize(train_emb, dim=1, p=2)
        query_emb = F.normalize(query_emb, dim=1, p=2)

        sim = torch.mm(train_emb, query_emb.T)
        ranks = torch.argsort(-sim, dim=0).cpu().numpy()

        mapM, mapH = train_loader.dataset.eval(ranks)

    if args.log:
        args.logger.log(
            f'dataset: {args.data}, map-M: {mapM:.1%}, map-H: {mapH:.1%}')
        
    print(f'Model: {args.modelname} CKPT: {args.ckptpath} Data: {args.data} map-M: {mapM:.1%}, map-H: {mapH:.1%}')


def main(args):

    tokenizer, model, dataloader = get_model_and_data(args)
    eval(model, tokenizer, dataloader, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
