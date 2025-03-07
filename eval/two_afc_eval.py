import torch as torch
import torch.nn.functional as F

from data.prompt_constants import internvl_prompt, qwen_prompt
from data import preprocess_prompt  # get_data
from eval.utils import get_c2s_score, parse_args, get_model_and_data  # , get_any_model, fix_seed
from perc_models.utils_perceptual_eval import get_acc


def eval_llava_next(_img0, _img1, _img2, args, input_ids, model, tokenizer):
    _img0, _img1, _img2 = _img0['pixel_values'].squeeze(0).to(args.device, dtype=model.dtype), \
        _img1['pixel_values'].squeeze(0).to(args.device, dtype=model.dtype), \
        _img2['pixel_values'].squeeze(0).to(args.device, dtype=model.dtype),
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids.unsqueeze(0).to(args.device),
            images=[_img0, _img1, _img2],
            image_sizes=224,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    outputs = outputs.split('.')[0]
    if ':' in outputs:
        outputs = outputs.split(':')[1]
    pred = 0 if ('A' in outputs or 'left' in outputs or '2' in outputs) else 1
    return pred


def eval_q_future(_img0, _img1, _img2, args, input_ids, model, tokenizer):
    # toks = ["inferior", "worse", "similar", "better", "superior"]
    toks = ["dissimilar", "different", "comparable", "similar", "identical"]
    _img0, _img1, _img2 = _img0['pixel_values'].squeeze(0).to(args.device, dtype=model.dtype), \
        _img1['pixel_values'].squeeze(0).to(args.device, dtype=model.dtype), \
        _img2['pixel_values'].squeeze(0).to(args.device, dtype=model.dtype),
    score_1 = get_c2s_score(_img0, _img1, toks, input_ids.unsqueeze(0).to(args.device), model, tokenizer)
    score_2 = get_c2s_score(_img0, _img2, toks, input_ids.unsqueeze(0).to(args.device), model, tokenizer)
    pred = 0 if score_1 > score_2 else 1
    outputs = f'{score_1}_{score_2}'
    return pred


def eval_mantis(_img0, _img1, _img2, args, input_ids, model, tokenizer):
    images = {
        'pixel_values': torch.cat((_img0['pixel_values'].squeeze(0), _img1['pixel_values'].squeeze(0),
                                   _img2['pixel_values'].squeeze(0)), dim=1).to(args.device,
                                                                                dtype=model.dtype),
        'pixel_attention_mask': torch.cat((_img0['pixel_attention_mask'].squeeze(0),
                                           _img1['pixel_attention_mask'].squeeze(0),
                                           _img2['pixel_attention_mask'].squeeze(0)), dim=1).to(args.device,
                                                                                                dtype=model.dtype)}
    input_ids = {k: v.to(args.device) for k, v in input_ids.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **input_ids,
            **images,
            do_sample=False,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True)
        output_ids = output_ids[:, input_ids["input_ids"].shape[1]:]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        outputs = outputs.split('.')[0]
        if ':' in outputs:
            outputs = outputs.split(':')[1]
        pred = 0 if ('A' in outputs or 'left' in outputs or '2' in outputs) else 1
    return input_ids, pred



def eval(model, tokenizer, test_loader, args):
    """
    Task: given are reference image and two alternatives, decide which is more similar
    to the reference.
    """

    print(f'Evaluating {args.data} dataset with length {len(test_loader)}.')

    if args.model_type == 'gen-lmmm':
        count, all = 0, 0

        with torch.no_grad():
            for i, (_img0, _img1, _img2, target, _) in enumerate(test_loader):

                if 'Mantis' in args.model_path:
                    input_ids = preprocess_prompt(args, tokenizer)
                    input_ids, pred = eval_mantis(_img0, _img1, _img2, args, input_ids, model, tokenizer)

                elif 'q-future' in args.model_path:
                    input_ids = preprocess_prompt(args, tokenizer)
                    pred = eval_q_future(_img0, _img1, _img2, args, input_ids, model, tokenizer)
                    
                elif 'InternVL2_5' in args.model_path:
                    generation_config = dict(max_new_tokens=1024, do_sample=True)
                    num_patches_list = [_img0.size(0), _img1.size(0), _img2.size(0)]
                    imgs = torch.concat([_img0, _img1, _img2], dim=0).to(args.device, dtype=model.dtype)
                    outputs = model.chat(tokenizer, imgs, internvl_prompt['2afc'], generation_config, num_patches_list=num_patches_list)
                    pred = 0 if ('A' in outputs or 'left' in outputs or '2' in outputs) else 1

                elif 'Qwen' in args.model_path:
                    text = tokenizer.apply_chat_template(
                                qwen_prompt['2afc'], tokenize=False, add_generation_prompt=True
                            )
                    inputs = tokenizer(
                        text=[text],
                        images=[_img0, _img1, _img2],
                        padding=True,
                        return_tensors="pt",
                    )
                    if i == 0:
                        print(text)
                        print(inputs)
                        print(inputs['input_ids'].shape, inputs['pixel_values'].shape)
                        print(_img0.size)
                    inputs = inputs.to("cuda")
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    outputs = tokenizer.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    pred = 0 if ('A' in outputs or 'left' in outputs or '2' in outputs) else 1
                    
                else:
                    input_ids = preprocess_prompt(args, tokenizer)
                    pred = eval_llava_next(_img0, _img1, _img2, args, input_ids, model, tokenizer)
                    
                count += pred == round(target)
                all += 1
                if all >= args.num_samples:
                    break
                print(f'dataset: {args.data}, clean accuracy: {count / all:.1%} ({count:.0f}/{all})', flush=True)


    elif args.model_type == 'perc-metric':
        assert args.metric_type == 'embedding'
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
            for e, (ref, img0, img1, target, _) in enumerate(test_loader):

                if e == 0:
                    print(img0.shape, target)
                ref_emb = _enc(ref.to(args.device))
                img0_emb = _enc(img0.to(args.device))
                img1_emb = _enc(img1.to(args.device))
                sim0 = F.cosine_similarity(img0_emb, ref_emb, dim=-1)
                sim1 = F.cosine_similarity(img1_emb, ref_emb, dim=-1)
                pred = sim0 < sim1
                count += (pred.long().to(target.device) == target.squeeze()).float().sum()
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
