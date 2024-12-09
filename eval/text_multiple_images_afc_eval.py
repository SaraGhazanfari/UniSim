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
        count = 0
        all = 0

        with torch.no_grad():
            while all < args.num_samples:
                for _, (imgs, lab, prompt) in enumerate(test_loader):
                    if 'Mantis' in args.model_path:
                        images = {
                            'pixel_values': torch.cat([img['pixel_values'].squeeze(0) for img in imgs], dim=1).to(args.device, 
                                                                                                                dtype=model.dtype),
                            'pixel_attention_mask': torch.cat([img['pixel_attention_mask'].squeeze(0) for img in imgs], 
                                                            dim=1).to(args.device, dtype=model.dtype)}
                        prompt = {k: v.squeeze(0).to(args.device) for k, v in prompt.items()}
                        output_ids = model.generate(
                            **prompt, **images,
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=1024,
                            use_cache=True)
                        output_ids = output_ids[:, prompt["input_ids"].shape[1]:]
                    else:
                        prompt, imgs, lab = prompt.to(args.device), \
                            [img['pixel_values'].squeeze(0).to(args.device, dtype=model.dtype) for img in imgs], \
                            lab.to(args.device)
                        with torch.inference_mode():
                            output_ids = model.generate(
                                prompt,
                                images=imgs,
                                image_sizes=224,
                                do_sample=True if args.temperature > 0 else False,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                num_beams=args.num_beams,
                                max_new_tokens=args.max_new_tokens,
                                use_cache=True)

                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    if 'A' in outputs or '1' in outputs:
                        pred = 0  
                    elif 'B' in outputs or '2' in outputs:
                        pred = 1
                    else:
                        pred = 2
                    count += float(pred == lab)
                    # print(e, outputs, '\t', pred, f'{count / all:.1%}')
                    all += 1
                    # print(
                    #     f'Target: {lab}, Output: "{outputs}", Pred: {pred}, Count: {count / all:.1%} ({count:.0f}/{all})')
                    if all >= args.num_samples:
                        break

        print(f'Data: {args.data}, Total Accuracy: {count / all:.1%} ({count}/{all})')

    elif (args.model_type == 'embedding-model' or
          args.modelname in ['liqe-mix', 'liqe-koniq']):
        # Get encoders.
        if args.modelname.startswith('dreamsim:'):
            _im_enc = model.embed
            _txt_enc = model.orig_clip.encode_text
        elif hasattr(model, 'encode_image'):
            _im_enc = model.encode_image
            _txt_enc = model.encode_text
        else:
            _im_enc = model.encode_image
            _txt_enc = model.encode_text

        all, count = 0, 0
        while all < args.num_samples:
            with torch.no_grad():  # torch.cuda.amp.autocast()
                for e, (imgs, lab, prompt) in enumerate(test_loader):
                    if e == 0 and all == 0:
                        print(len(imgs), imgs[0].shape, imgs[0][0, :, 0, :2], lab)
                    tks = tokenizer(prompt)
                    txt_emb = _txt_enc(tks.to(args.device))
                    imgs_emb = [_im_enc(_img.to(args.device)) for _img in imgs]
                    sims = [F.cosine_similarity(_img_emb, txt_emb, dim=-1) for _img_emb in imgs_emb]
                    pred = torch.stack(sims, dim=-1).argmax(dim=-1)  # Predict the most similar image (0 --> `img0` is more similar)
                    count += (pred.long().to(lab.device) == lab).float().sum()
                    all += imgs[0].shape[0]

                    print(f'Clean accuracy: {count / all:.1%} ({count:.0f}/{all})')

                    if all >= args.num_samples:
                        break

    elif args.model_type == 'score-model':
        all, count = 0, 0
        while all < args.num_samples:
            with torch.no_grad():  # torch.cuda.amp.autocast()
                for e, (imgs, lab, prompt) in enumerate(test_loader):
                    if e == 0 and all == 0:
                        print(len(imgs), imgs[0].shape, imgs[0][0, :, 0, :2])
                    tks = tokenizer(prompt)
                    scores = [model.score(tks.to(args.device), _img.to(args.device)) for _img in imgs]
                    pred = torch.stack(scores, dim=-1).argmax(dim=-1)  # Predict image with highest score.
                    count += (pred.long().to(lab.device) == lab).float().sum()
                    all += imgs[0].shape[0]

                    print(f'Clean accuracy: {count / all:.1%} ({count:.0f}/{all})')

                    if all >= args.num_samples:
                        break

    if args.log:
        args.logger.log(
            f'dataset: {args.data}, clean accuracy: {count / all:.1%} ({count:.0f}/{all})')


def main(args):
    
    tokenizer, model, dataloader = get_model_and_data(args)
    eval(model, tokenizer, dataloader, args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    args = parse_args()
    main(args)
