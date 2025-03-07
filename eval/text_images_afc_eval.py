import torch as torch
import torch.nn.functional as F
#from torch.utils.data import DataLoader

from data.prompt_constants import qwen_prompt, internvl_prompt
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
        while all < args.num_samples:
            with torch.no_grad():
                for idx, (img0, img1, lab, prompt) in enumerate(test_loader):
                    if 'Mantis' in args.model_path:
                        images = {
                            'pixel_values': torch.cat(
                                (img0['pixel_values'].squeeze(0), img1['pixel_values'].squeeze(0)), dim=1).to(
                                args.device, dtype=model.dtype),
                            'pixel_attention_mask': torch.cat((img0['pixel_attention_mask'].squeeze(0),
                                                            img1['pixel_attention_mask'].squeeze(0),
                                                            ), dim=1).to(args.device, dtype=model.dtype)}
                        prompt = {k: v.squeeze(0).to(args.device) for k, v in prompt.items()}
                        output_ids = model.generate(
                            **prompt, **images,
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=1024,
                            use_cache=True)
                        output_ids = output_ids[:, prompt["input_ids"].shape[1]:]
                        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                        
                    elif 'InternVL2_5' in args.model_path:
                        generation_config = dict(max_new_tokens=1024, do_sample=True)
                        num_patches_list = [img0.size(0), img1.size(0)]
                        imgs = torch.concat([img0, img1], dim=0).to(args.device, dtype=model.dtype)
                        outputs = model.chat(tokenizer, imgs, internvl_prompt['text_images_afc'].format(prompt=prompt),
                                            generation_config, num_patches_list=num_patches_list)

                    elif 'Qwen' in args.model_path:
                        temp_prompt = qwen_prompt['text_images_afc']
                        temp_prompt[0]['content'][3]['text'] = temp_prompt[0]['content'][3]['text'].format(prompt=prompt)

                        text = tokenizer.apply_chat_template(
                                    temp_prompt, tokenize=False, add_generation_prompt=True
                                )
                        inputs = tokenizer(
                            text=[text],
                            images=[img0, img1],
                            padding=True,
                            return_tensors="pt",
                        )
                        if idx == 0:
                            print(text)
                            print(inputs)
                            print(inputs['input_ids'].shape, inputs['pixel_values'].shape)
                            print(img0.size)
                        inputs = inputs.to("cuda")
                        generated_ids = model.generate(**inputs, max_new_tokens=128)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        outputs = tokenizer.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                    else:
                        prompt, img0, img1 = prompt.to(args.device), \
                            img0['pixel_values'].squeeze(0).to(args.device, dtype=model.dtype), \
                            img1['pixel_values'].squeeze(0).to(args.device, dtype=model.dtype)
                        with torch.inference_mode():
                            output_ids = model.generate(
                                prompt.unsqueeze(0),
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

                    print(f'dataset: {args.data}, clean accuracy: {count / all:.1%} ({count:.0f}/{all})', flush=True)
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
                for e, (img0, img1, lab, prompt) in enumerate(test_loader):

                    if e == 0 and count == 0:
                        print(img0.shape, lab)
                    tks = tokenizer(prompt)
                    txt_emb = _txt_enc(tks.to(args.device))
                    img0_emb = _im_enc(img0.to(args.device))
                    img1_emb = _im_enc(img1.to(args.device))
                    # if args.modelname != 'hps':
                    sim0 = F.cosine_similarity(img0_emb, txt_emb, dim=-1)
                    sim1 = F.cosine_similarity(img1_emb, txt_emb, dim=-1)
                    # else:
                    #     sim0 = (img0_emb * txt_emb).sum(-1)
                    #     sim1 = (img1_emb * txt_emb).sum(-1)
                    pred = sim0 < sim1  # Predict the most similar image (0 --> `img0` is more similar)
                    count += (pred.long().to(lab.device) == lab).float().sum()
                    all += img0.shape[0]
                    
                    if all >= args.num_samples:
                        break
                    

    elif args.model_type == 'score-model':
        all, count = 0, 0
        while all < args.num_samples:
            with torch.no_grad():  # torch.cuda.amp.autocast()
                for e, (img0, img1, lab, prompt) in enumerate(test_loader):
                    
                    if e == 0 and count == 0:
                        print(img0.shape, lab)
                    tks = tokenizer(prompt)
                    scores0 = model.score(tks.to(args.device), img0.to(args.device))
                    scores1 = model.score(tks.to(args.device), img1.to(args.device))
                    pred = scores0 < scores1  # Predict image with highest score.
                    count += (pred.long().to(lab.device) == lab).float().sum()
                    all += img0.shape[0]

                    if all >= args.num_samples:
                        break

    if args.log:
        args.logger.log(
            f'dataset: {args.data}, clean accuracy: {count / all:.1%} ({count:.0f}/{all})')
        
    print(f'Model: {args.model_path} CKPT: {args.ckptpath} Data: {args.data} Total Accuracy: {count / all:.1%} ({count}/{all})')



def main(args):
    
    tokenizer, model, dataloader = get_model_and_data(args)
    eval(model, tokenizer, dataloader, args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    args = parse_args()
    main(args)
