import torch as torch
import torch.nn.functional as F

from eval.utils import parse_args, get_model_and_data


def eval(model, tokenizer, test_loader, args):
    """
    Task: given an image and two descriptions, decide which one better describes
    the image.
    """

    print(f'Evaluating {args.data} dataset.')

    if args.model_type == 'gen-lmmm':
        count, all = 0, 0
        while all < args.num_samples:
            with torch.no_grad():
                for _, (img0, prompt, lab) in enumerate(test_loader):
                    if 'Mantis' in args.model_path:
                        img0['pixel_values'] = img0['pixel_values'].squeeze(0).to(device=args.device, dtype=model.dtype)
                        img0['pixel_attention_mask'] = img0['pixel_attention_mask'].squeeze(0).to(device=args.device,
                                                                                                dtype=model.dtype)
                        prompt = {k: v.squeeze(0).to(args.device) for k, v in prompt.items()}
                        output_ids = model.generate(
                            **prompt, **img0,
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=1024,
                            use_cache=True)
                        output_ids = output_ids[:, prompt["input_ids"].shape[1]:]
                    else:
                        prompt, img0, lab = prompt.to(args.device), \
                            img0['pixel_values'].squeeze(0).to(args.device, dtype=model.dtype), lab.to(args.device)
                        with torch.inference_mode():
                            output_ids = model.generate(
                                prompt,
                                images=[img0],
                                image_sizes=224,
                                do_sample=True if args.temperature > 0 else False,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                num_beams=args.num_beams,
                                max_new_tokens=args.max_new_tokens,
                                use_cache=True)

                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    pred = 0 if ('A' in outputs or 'left' in outputs or '1' in outputs) else 1
                    count += float(pred == lab)
                   
                    all += 1
                    if all >= args.num_samples:
                        break


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
            _im_enc = model.get_image_features
            _txt_enc = model.get_text_features
        
        all, count = 0, 0
        while all < args.num_samples:
            with torch.no_grad():  # torch.cuda.amp.autocast()
                for e, (img, txt0, txt1, lab) in enumerate(test_loader):
                    if e == 0 and all == 0:
                        print(img.shape, img.max(), img.min())
                    tks0 = tokenizer(txt0)
                    tks1 = tokenizer(txt1)
                    txt0_emb = _txt_enc(tks0.to(args.device))
                    txt1_emb = _txt_enc(tks1.to(args.device))
                    img_emb = _im_enc(img.to(args.device))
                    sim0 = F.cosine_similarity(img_emb, txt0_emb, dim=-1)
                    sim1 = F.cosine_similarity(img_emb, txt1_emb, dim=-1)
                    pred = sim0 < sim1  # Predict the most similar image (0 --> `img0` is more similar)
                    count += (pred.long().to(lab.device) == lab).float().sum()
                    all += img.shape[0]


                    if all >= args.num_samples:
                        break
                    
    elif args.model_type == 'score-model':
        all, count = 0, 0
        while all < args.num_samples:
            with torch.no_grad():  # torch.cuda.amp.autocast()
                for e, (img, txt0, txt1, lab) in enumerate(test_loader):
                    tks0 = tokenizer(txt0)
                    tks1 = tokenizer(txt1)
                    scores0 = model.score(tks0.to(args.device), img.to(args.device))
                    scores1 = model.score(tks1.to(args.device), img.to(args.device))
                    pred = scores0 < scores1  # Predict image with highest score.
                    count += (pred.long().to(lab.device) == lab).float().sum()
                    all += img.shape[0]

                    if all >= args.num_samples:
                        break

    if args.log:
        args.logger.log(
            f'dataset: {args.data}, clean accuracy: {count / all:.1%} ({count:.0f}/{all})')
    print(f'Model: {args.modelname} CKPT: {args.ckptpath} Data: {args.data} Total Accuracy: {count / all:.1%} ({count}/{all})')



def main(args):
    import warnings

    warnings.filterwarnings("ignore")
    tokenizer, model, dataloader = get_model_and_data(args)
    eval(model, tokenizer, dataloader, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
