import os
import torch
from PIL import Image
from torch.nn import functional as F
import copy


def resolve_ckpt(ckpt):
    if 'state_dict' in ckpt.keys():
        ckpt = ckpt['state_dict']
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    return ckpt


def load_unisim_models(model_name, model_path, device='cuda:0', cache_dir='./'):
    clip_model_dict = {
        'unisim_vit_b_32':'laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
        'unisim_vit_l_14':'openai/clip-vit-large-patch14-336'
    }
    if model_name in ['unisim_vit_b_32', 'unisim_vit_l_14']:
        from train.open_clip_train.main import UniSim
        from types import SimpleNamespace
        args= { 
                'model':  clip_model_dict[model_name],
                'lora': True,
                'lora_r': 16,                
                'lora_alpha': 32,    
                'lora_dropout': 0.2}
        
        args = SimpleNamespace(**args)
        model = UniSim(args, device=device)
        print(f'Loading checkpoint from {model_path}.')
        ckpt = torch.load(model_path, map_location='cpu')
        ckpt = resolve_ckpt(ckpt)
        model.load_state_dict(ckpt, strict=True)    
        model.to(device)
        model.eval()    
        return model.tokenizer_val, model, model.preprocess_val
        
    if model_name == 'unisim_ll_n_0.5':
        from models.llava_next.model.builder import load_pretrained_model
        model_path = os.path.expanduser(model_path)
        model_name = 'llava_qwen'
        tokenizer, model, image_processor, _ = load_pretrained_model(
                model_path,
                None,
                model_name,
                device_map='auto',
                use_cache=True,
                cache_dir=cache_dir)
        
        return tokenizer, model, image_processor 
    else:
        raise Exception('Selected model is not supported! Choose from unisim_vit_b_32, unisim_vit_l_14 and unisim_ll_n_0.5')
        
def get_unisim_encoder_metric(model, task_type, image_processor, tokenizer, images, texts):
        image_embeds = model.encode_image(torch.stack([image_processor(Image.open(img_), 
                                                                       return_tensors='pt') for img_ in images]).to(device))
        if len(texts) > 0:
            texts_embeds = model.encode_text(torch.stack([tokenizer(text_) for text_ in texts]).to(device))
            
        if task_type == 'Img_2AFC':
            assert len(images) == 3 and len(texts) == 0
            sim_0 = F.cosine_similarity(image_embeds[0].unsqueeze(0), image_embeds[1].unsqueeze(0))
            sim_1 = F.cosine_similarity(image_embeds[2].unsqueeze(0), image_embeds[1].unsqueeze(0))
            pred = (sim_0 < sim_1).sum()
            
        elif task_type == 'OOO':
            assert len(images) == 3 and len(texts) == 0
            sim_01 = F.cosine_similarity(image_embeds[0].unsqueeze(0), image_embeds[1].unsqueeze(0))
            sim_12 = F.cosine_similarity(image_embeds[1].unsqueeze(0), image_embeds[2].unsqueeze(0))
            sim_02 = F.cosine_similarity(image_embeds[0].unsqueeze(0), image_embeds[1].unsqueeze(0))
            pred = torch.cat(sim_12, sim_02, sim_01).argmin()
        
        elif task_type == 'Text_2AFC':
            assert len(images) == 1 and len(texts) == 2
            sim_0 = F.cosine_similarity(image_embeds[0].unsqueeze(0), texts_embeds[0].unsqueeze(0))
            sim_1 = F.cosine_similarity(image_embeds[0].unsqueeze(0), texts_embeds[1].unsqueeze(0))
            pred = (sim_0 < sim_1).sum()
        
        elif task_type in ['IT_AFC', 'IQA', 'PAA']:
            assert len(images) == 2 and len(texts) == 1
            sim_0 = F.cosine_similarity(image_embeds[0].unsqueeze(0), texts_embeds[0].unsqueeze(0))
            sim_1 = F.cosine_similarity(image_embeds[1].unsqueeze(0), texts_embeds[0].unsqueeze(0))
            pred = (sim_0 < sim_1).sum()
 
 
def preprocess_text(tokenizer, prompt):
    from models.llava_next.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from models.llava_next.conversation import conv_templates
    from models.llava_next.mm_utils import tokenizer_image_token
    conv_template = "qwen_1_5"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    return input_ids


def get_unisim_lmm_metric(model, task_type, image_processor, tokenizer, images, texts):
    from data.prompt_constants import llava_next_prompt  
    TASK_TYPE = {'Img_2AFC': '2afc', 'OOO':'3afc', 
                 'IT_AFC': 'text_images_afc', 'IQA': 'iqa' ,
                 'Text_2AFC': 'text_2afc', 'PAA': 'paa'} 
    task_type = TASK_TYPE[task_type]
    
    if task_type == 'paa':
        if 'color' in texts[0]:
            task_type = 'colorfulness'
        elif 'bright' in texts[0]:
            task_type = 'brightness'
        elif 'contrast' in texts[0]:
            task_type = 'contrast'
        elif 'sharp' in texts[0]:
            task_type = 'sharpness'
            
    prompt = llava_next_prompt[task_type]
    images = [image_processor.preprocess(Image.open(img_), 
                              return_tensors='pt')['pixel_values'].squeeze(0).to(device, 
                              dtype=model.dtype) for img_ in images]  
        
    if task_type == 'text_2afc':
        prompt =  prompt.format(cap1=texts[0], cap2=texts[1])
        
    elif task_type == 'text_images_afc':
        prompt = prompt.format(prompt=texts[0])
        
    
    input_ids = preprocess_text(tokenizer, prompt)
    output_ids = model.generate(
        input_ids.to(device).unsqueeze(0),
        images=images,
        image_sizes=224,
        do_sample=False,
        temperature=0,
        top_p=1,
        num_beams=1,
        max_new_tokens=512,
        use_cache=True)                    
                           
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    pred = 0 if ('A' in outputs or 'left' in outputs or '1' in outputs) else 1
    return pred

    
def get_unisim_metric(model_name, model_path, task_type, images=list(), texts=list(), 
                      device='cuda:0', cache_dir='./'):
    
    
    tokenizer, model, image_processor = load_unisim_models(model_name, model_path, device, cache_dir)
    
    if model_name in ['unisim_vit_b_32', 'unisim_vit_l_14']:
        pred = get_unisim_encoder_metric(model, task_type, image_processor, tokenizer, images, texts)
    else:
        pred = get_unisim_lmm_metric(model, task_type, image_processor, tokenizer, images, texts)
    return pred
    

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    model_name = 'unisim_ll_n_0.5'
    model_path = 'path/to/model'
    device='cuda:0'
    cache_dir='./'
    
    images = ['/uni_data/nights/distort/000/002_0.png', '/uni_data/nights/ref/000/002.png', 
              '/uni_data/nights/distort/000/002_1.png']
    texts = []
    task_type = 'Img_2AFC'
    pred = get_unisim_metric(model_name, model_path, task_type, images, texts, 
                      device, cache_dir)
    
    print(f'Task: {task_type}, Pred: {pred}')
    
    
    texts = ["make the cat's nose black"]
    images = ["/uni_data/MagicBrush/test/1980.jpg", "/uni_data/MagicBrush/test/1979.jpg"]
    task_type = 'IT_AFC'
    pred = get_unisim_metric(model_name, model_path, task_type, images, texts, 
                      device, cache_dir)
    
    print(f'Task: {task_type}, Pred: {pred}')
    
    