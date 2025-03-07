import torch as torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.prompt_constants import internvl_prompt, qwen_prompt
from data import preprocess_prompt  # get_data
from eval.utils import parse_args, get_model_and_data #get_any_model
from perc_models.utils_perceptual_eval import get_acc


def eval(model, tokenizer, test_loader, args):
    """
    Task: find the odd-one-out of three images.
    """
    
    print(f'Evaluating {args.data} dataset.')

    if args.model_type == 'gen-lmmm':
        count = 0
        all = 0
        device = args.device
        
        with torch.no_grad():
            for e, (_img0, _img1, _img2, lab, idx) in enumerate(test_loader):
                if 'Mantis' in args.model_path:
                    input_ids = preprocess_prompt(args, tokenizer)
                    images = {
                        'pixel_values': torch.cat(
                            (_img0['pixel_values'].squeeze(0), _img1['pixel_values'].squeeze(0),
                             _img2['pixel_values'].squeeze(0)), dim=1).to(
                            args.device, dtype=model.dtype),
                        'pixel_attention_mask': torch.cat((_img0['pixel_attention_mask'].squeeze(0),
                                                           _img1['pixel_attention_mask'].squeeze(0),
                                                           _img2['pixel_attention_mask'].squeeze(0)), dim=1).to(
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
                    
                elif 'InternVL2_5' in args.model_path:
                    generation_config = dict(max_new_tokens=1024, do_sample=True)
                    num_patches_list = [_img0.size(0), _img1.size(0), _img2.size(0)]
                    imgs = torch.concat([_img0, _img1, _img2], dim=0).to(args.device, dtype=model.dtype)
                    outputs = model.chat(tokenizer, imgs, internvl_prompt['3afc'], generation_config, num_patches_list=num_patches_list)


                elif 'Qwen' in args.model_path:
                    try:
                        text = tokenizer.apply_chat_template(
                                    qwen_prompt['3afc'], tokenize=False, add_generation_prompt=True
                                )
                        inputs = tokenizer(
                            text=[text],
                            images=[_img0, _img1, _img2],
                            padding=True,
                            return_tensors="pt",
                        )
                        if e == 0:
                            print(text)

                        inputs = inputs.to("cuda")
                        generated_ids = model.generate(**inputs, max_new_tokens=128)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        outputs = tokenizer.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                    except Exception as e:
                        print(e)
                else:
                    input_ids = preprocess_prompt(args, tokenizer)
                    _img0, _img1, _img2 = _img0['pixel_values'].squeeze(0).to(device, dtype=model.dtype), \
                        _img1['pixel_values'].squeeze(0).to(device, dtype=model.dtype), \
                        _img2['pixel_values'].squeeze(0).to(device, dtype=model.dtype)
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids.to(device).unsqueeze(0),
                            images=[_img0, _img1, _img2],
                            image_sizes=384,
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True)

                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                answ = outputs.split('.')[0]
                answ = answ.replace('Answer: ', '')
                pred = 'A' if 'A' in answ or '1' in answ else 'B' if 'B' in answ or '2' in answ else 'C'
                target = ['A', 'B', 'C'][lab]
                count += float(pred == target)
                all += 1
                print(f'dataset: {args.data}, clean accuracy: {count / all:.1%} ({count:.0f}/{all})', flush=True)
                if all >= args.num_samples:
                    break


    elif args.model_type == 'perc-metric':
        assert args.metric_type == 'odd-one-out'
        with torch.no_grad():
            acc, count, all = get_acc(
                model, test_loader, device=args.device, n_ex=args.n_ex,
                mode=args.metric_type, verbose=True)
        

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

        count, all = 0, 0
        with torch.no_grad():
            for e, (img0, img1, img2, target, _) in enumerate(test_loader):

                if e == 0:
                    print(img0.shape, target)
                img0_emb = _enc(img0.to(args.device))
                img1_emb = _enc(img1.to(args.device))
                img2_emb = _enc(img2.to(args.device))
                sim01 = F.cosine_similarity(img0_emb, img1_emb, dim=-1)
                sim02 = F.cosine_similarity(img0_emb, img2_emb, dim=-1)
                sim12 = F.cosine_similarity(img1_emb, img2_emb, dim=-1)
                pred = torch.stack((sim12, sim02, sim01), dim=1).argmax(dim=1)
                count += (pred.long().to(target.device) == target).float().sum()
                all += img0.shape[0]

               

                if all >= args.num_samples:
                    break

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
