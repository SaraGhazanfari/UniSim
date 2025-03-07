import torch as torch
import torch.nn.functional as F
#from torch.utils.data import DataLoader

from data import get_data, preprocess_prompt
from data.prompt_constants import LONG_STRING, ATTRIBUTE_PROMPTS, internvl_prompt, qwen_prompt
from eval.utils import parse_args, get_model_and_data


def eval(model, tokenizer, test_loader, args):
    """
    Task: given two images, decide which one has higher quality.
    """

    print(f'Evaluating {args.data} dataset.')

    def _get_prompt(args):
        """Return prompts to use."""
 
        if args.mode == 'naive-rev':
            prompt = LONG_STRING
        
        elif args.mode == 'naive':
            attr = getattr(args, 'attribute', 'quality')  # In case other attributes are used.
            prompt = ATTRIBUTE_PROMPTS['naive'][attr]
        
        elif args.mode == 'clip-iqa':
            prompt = ('Good photo.', 'Bad photo.')
            attr = getattr(args, 'attribute', None)  # For KonIQ10k.
            if attr:
                prompt = ATTRIBUTE_PROMPTS['clip-iqa'][attr]
            
        print(prompt)
        return prompt
        

    if args.model_type == 'gen-lmmm':

        count = 0
        all = 0
        print(args.num_samples)
        while all < args.num_samples:
            for idx in range(len(test_loader)):
                img0, img1, lab, _ = test_loader.__getitem__(idx)
                if 'Mantis' in args.model_path:
                    input_ids = preprocess_prompt(args, tokenizer)
                    images = {
                        'pixel_values': torch.cat(
                            (img0['pixel_values'].squeeze(0), img1['pixel_values'].squeeze(0)), dim=1).to(
                            args.device, dtype=model.dtype),
                        'pixel_attention_mask': torch.cat((img0['pixel_attention_mask'].squeeze(0),
                                                        img1['pixel_attention_mask'].squeeze(0)), dim=1).to(
                            args.device,
                            dtype=model.dtype)}
                    input_ids = {k: v.to(args.device) for k, v in input_ids.items()}
                    output_ids = model.generate(
                        **input_ids, **images,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=1024,
                        use_cache=True)
                    output_ids = output_ids[:, input_ids["input_ids"].shape[1]:]
                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    pred = 0 if ('A' in outputs or 'left' in outputs or '1' in outputs) else 1

                elif 'q-future' in args.model_path:
                    from PIL import Image
                    score_0 = model.score(Image.open(img0[0]).convert('RGB'))
                    score_1 = model.score(Image.open(img1[0]).convert('RGB'))
                    pred = score_0 < score_1

                elif 'Qwen' in args.model_path:
                    text = tokenizer.apply_chat_template(
                        qwen_prompt['iqa'], tokenize=False, add_generation_prompt=True
                    )

                    inputs = tokenizer(
                        text=[text],
                        images=[img0, img1],
                        padding=True,
                        return_tensors="pt",
                    )
                    if idx == 0:
                        print(inputs['pixel_values'].shape)
                    inputs = inputs.to("cuda")
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    outputs = tokenizer.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    pred = 0 if ('A' in outputs or 'left' in outputs or '1' in outputs) else 1

                elif 'InternVL2_5' in args.model_path:
                    generation_config = dict(max_new_tokens=1024, do_sample=True)
                    num_patches_list = [img0.size(0), img1.size(0)]
                    imgs = torch.concat([img0, img1], dim=0).to(args.device, dtype=model.dtype)
                    outputs = model.chat(tokenizer, imgs, internvl_prompt['iqa'],
                                        generation_config, num_patches_list=num_patches_list)
                    pred = 0 if ('A' in outputs or 'left' in outputs or '1' in outputs) else 1
                    
                else:
                    input_ids = preprocess_prompt(args, tokenizer)
                    img0, img1 = img0['pixel_values'].squeeze(0).to(args.device, dtype=model.dtype), \
                        img1['pixel_values'].squeeze(0).to(args.device, dtype=model.dtype),
                    output_ids = model.generate(
                        input_ids.to(args.device).unsqueeze(0),
                        images=[img0, img1],
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
                print(f'dataset: {args.data}, clean accuracy: {count / all:.1%} ({count:.0f}/{all})')
                if all >= args.num_samples:
                    break


    elif (args.model_type == 'embedding-model' or
          args.modelname in ['liqe-mix', 'liqe-koniq'] and 
          getattr(args, 'attribute', 'quality') != 'quality'):
        if args.modelname in ['liqe-mix', 'liqe-koniq']: 
          print(f"Warning: using embedding models evalution for LIQE,"
                f" it should be used only for PAA."
                f"Current attribute: {getattr(args, 'attribute', 'quality')}."
          )
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
        with torch.no_grad():  # torch.cuda.amp.autocast()

            if args.mode in ['naive', 'naive-rev', 'naive-prompt']:
                prompt = _get_prompt(args)
                tks = tokenizer(prompt)
                txt_emb = _txt_enc(tks.to(args.device))
                while all < args.num_samples:
                    for e, (img0, img1, lab, _) in enumerate(test_loader):
                        if e == 0 and all == 0:
                            print(img0.shape, img0.max(), img0.min(), lab)
                        img0_emb = _im_enc(img0.to(args.device))
                        img1_emb = _im_enc(img1.to(args.device))
                        sim0 = F.cosine_similarity(img0_emb, txt_emb, dim=-1)
                        sim1 = F.cosine_similarity(img1_emb, txt_emb, dim=-1)
                        if args.mode == 'naive':
                            pred = sim0 < sim1  # Predict the most similar image (0 --> `img0` is more similar)
                        elif args.mode == 'naive-rev':
                            pred = sim0 > sim1
                        count += (pred.long().to(lab.device) == lab).float().sum()
                        all += img0.shape[0]

                        print(f'Clean accuracy: {count / all:.1%} ({count:.0f}/{all})')

                        if all >= args.num_samples:
                            break
                print(f'Data: {args.data}, Clean accuracy: {count / all:.1%} ({count:.0f}/{all})')

            elif args.mode == 'clip-iqa':  # Approach adapted from https://arxiv.org/abs/2207.12396.
                prompts = _get_prompt(args)
                print(prompts)
                tks = tokenizer(prompts)
                txt_emb = _txt_enc(tks.to(args.device))
                while all < args.num_samples:
                    for e, (img0, img1, lab, _) in enumerate(test_loader):
                        if e == 0 and all == 0:
                            print(img0.shape, #img0.max(), img0.min(), img0[0, :, 0, 0],
                                    lab)
                        img0_emb = _im_enc(img0.squeeze(0).to(args.device))
                        img1_emb = _im_enc(img1.squeeze(0).to(args.device))
                        sim0 = torch.stack(
                            (F.cosine_similarity(img0_emb, txt_emb[0].unsqueeze(0), dim=-1),
                            F.cosine_similarity(img0_emb, txt_emb[1].unsqueeze(0), dim=-1),
                            ), dim=1)  # [sim(img0, 'good photo'), sim(img0, 'bad photo')]
                        sim0 = F.softmax(sim0, dim=-1)[:, 0]
                        sim1 = torch.stack(
                            (F.cosine_similarity(img1_emb, txt_emb[0].unsqueeze(0), dim=-1),
                            F.cosine_similarity(img1_emb, txt_emb[1].unsqueeze(0), dim=-1),
                            ), dim=1)
                        sim1 = F.softmax(sim1, dim=-1)[:, 0]
                        pred = sim0 < sim1  # Predict the most similar image (0 --> `img0` is more similar)
                        count += (pred.long().to(lab.device) == lab).float().sum()
                        all += img0.shape[0]


                        if all >= args.num_samples:
                            break
                print(f'Data: {args.data}, Clean accuracy: {count / all:.1%} ({count:.0f}/{all})')
        
            else:
                raise ValueError(f'Unknown mode: {args.mode}.')

    elif args.model_type == 'score-model':
        all, count = 0, 0
        if args.modelname in ['liqe-mix', 'liqe-koniq']:
            args.mode = 'image-only'

        if args.mode in ['naive', 'naive-rev', 'naive-prompt']:
            prompt = _get_prompt(args)
            tks = tokenizer([prompt] * args.batch_size).to(args.device)
            with torch.no_grad():
                while all < args.num_samples:
                    for _, (img0, img1, lab, _) in enumerate(test_loader):
                        sh = img0.shape[0]  # For last batch.
                        sim0 = model.score(image=img0.to(args.device), prompt=tks[:sh])
                        sim1 = model.score(image=img1.to(args.device), prompt=tks[:sh])
                        if args.mode == 'naive':
                            pred = sim0 < sim1  # Predict the most similar image (0 --> `img0` is more similar)
                        elif args.mode == 'naive-rev':
                            pred = sim0 > sim1
                        count += (pred.long().to(lab.device) == lab).float().sum()
                        all += img0.shape[0]

                        print(f'Clean accuracy: {count / all:.1%} ({count:.0f}/{all})', flush=True)

                        if all >= args.num_samples:
                            break
            print(f'Data: {args.data}, Clean accuracy: {count / all:.1%} ({count:.0f}/{all})')

        elif args.mode == 'clip-iqa':  # Approach adapted from https://arxiv.org/abs/2207.12396.
            prompts = _get_prompt(args)
            tks1 = tokenizer([prompts[0]] * args.batch_size).to(args.device)
            tks2 = tokenizer([prompts[1]] * args.batch_size).to(args.device)
            with torch.no_grad():
                while all < args.num_samples:
                    for e, (img0, img1, lab, _) in enumerate(test_loader):
                        sh = img0.shape[0]
                        sim0 = torch.stack((
                            model.score(image=img0.to(args.device), prompt=tks1[:sh]), 
                            model.score(image=img0.to(args.device), prompt=tks2[:sh])), dim=1)
                        sim0 = F.softmax(sim0, dim=-1)[:, 0]
                        sim1 = torch.stack((
                            model.score(image=img1.to(args.device), prompt=tks1[:sh]),
                            model.score(image=img1.to(args.device), prompt=tks2[:sh])), dim=1)
                        sim1 = F.softmax(sim1, dim=-1)[:, 0]
                        pred = sim0 < sim1  # Predict the most similar image (0 --> `img0` is more similar)
                        count += (pred.long().to(lab.device) == lab).float().sum()
                        all += img0.shape[0]

                        print(f'Clean accuracy: {count / all:.1%} ({count:.0f}/{all})')

                        if all >= args.num_samples:
                            break
                print(f'Data: {args.data}, Clean accuracy: {count / all:.1%} ({count:.0f}/{all})')

        elif args.mode == 'image-only':
            with torch.no_grad():
                while all < args.num_samples:
                    for e, (img0, img1, lab, _) in enumerate(test_loader):
                        if e == 0 and all == 0:
                            print(img0.shape)
                        score0 = model(img0.to(args.device))
                        score1 = model(img1.to(args.device))
                        pred = score0 < score1  # Predict the image highest score.
                        count += (pred.long().to(lab.device) == lab).float().sum()
                        all += img0.shape[0]

                        print(f'Clean accuracy: {count / all:.1%} ({count:.0f}/{all})')

                        if all >= args.num_samples:
                            break
            print(f'Data: {args.data}, Clean accuracy: {count / all:.1%} ({count:.0f}/{all})')

        else:
            raise ValueError(f'Unknown mode: {args.mode}.')
        
    if args.log:
        args.logger.log(
            f'dataset: {args.data}, clean accuracy: {count / all:.1%} ({count:.0f}/{all})')
    
    print(f'Model: {args.model_path} CKPT: {args.ckptpath} Data: {args.data} Total Accuracy: {count / all:.1%} ({count}/{all})')

def main(args):
    import warnings

    warnings.filterwarnings("ignore")
    tokenizer, model, dataloader = get_model_and_data(args)
    eval(model, tokenizer, dataloader, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
