from eval.utils import get_model_and_data, parse_args


def main(args):
    import warnings

    warnings.filterwarnings("ignore")
    tokenizer, model, dataloader = get_model_and_data(args)
    
    
    if args.data in ['night', 'bapps', 'pie-app-2afc']:
        from eval import two_afc_eval
        two_afc_eval.eval(model, tokenizer, dataloader, args)
        
    elif args.data in ['h-imagenet-triplets', 'cifar100coarse-triplets']:
        from eval import odd_one_out_eval
        odd_one_out_eval.eval(model, tokenizer, dataloader, args)
        
    elif args.data in ['kadid-pairs', 'koniq-pairs', 'pie-app-iqa', 'agiqa3k-qual-pairs', 'pipal']:
        from eval import image_quality_eval
        image_quality_eval.eval(model, tokenizer, dataloader, args)
    
    elif args.data in ['sice-pairs-h', 'koniq-brightness-pairs', 'koniq-colorfulness-pairs', 'koniq-contrast-pairs', 
                       'koniq-sharpness-pairs']:
        from eval import image_quality_eval
        image_quality_eval.eval(model, tokenizer, dataloader, args)
                  
    elif args.data in ['imagereward1k-pairs', 'hq_edit_text_images', 'agiqa3k-align-pairs', 'hpdv2-pairs', 
                       'magic_brush' , 'hpdv2-mindist-pairs']:
        from eval import text_images_afc_eval
        text_images_afc_eval.eval(model, tokenizer, dataloader, args)
    
    elif args.data in ['hq_edit_text_2afc', 'polaris', 'coco-triplets']:
        from eval import text_2afc_eval
        text_2afc_eval.eval(model, tokenizer, dataloader, args)
    
    elif args.data in ['rparis6k', 'roxford5k']:
        from eval import i2i_retrieval_eval
        i2i_retrieval_eval.eval(model, tokenizer, dataloader, args)
        


if __name__ == "__main__":
    args = parse_args()
    main(args)