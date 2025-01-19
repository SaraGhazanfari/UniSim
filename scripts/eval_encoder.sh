CACHE_DIR='all_models/'
DATA_DIR='/uni_data'
MODEL_DIR='./'


evaluate() {
    local models=("${!1}")   
    local ckpts=("${!2}")
    local tasks=("${!3}")

    for model in "${models[@]}"
    do
        for ckpt in "${ckpts[@]}"
        do
            for task in "${tasks[@]}"
            do
                $SCRATCH/pytorch-example/python -m $task \
                    --cache_dir $CACHE_DIR \
                    --data-path $DATA_DIR \
                    --model_dir $MODEL_DIR \
                    --modelname $model \
                    \
                    --attn_implementation 'sdpa' \
                    --temperature 0 \
                    --num-workers 8 \
                    \
                    \
                    --seed 0 \
                    $ckpt 
            done
        done
    done
}

declare -a tasks=(#

                "eval.total_eval --data night --split test --num-samples 1824 --batch-size 50"
                "eval.total_eval --data bapps --split val --num-samples 5000 --batch-size 50"
                "eval.total_eval --data pie-app-2afc --split test --num-samples 3314 --batch-size 50"
                    
                "eval.total_eval --data imagereward1k-pairs --split validation --num-samples 412 --batch-size 50"
                "eval.total_eval --data hpdv2-pairs --num-samples 5000 --batch-size 50 --split test"
                "eval.total_eval --data agiqa3k-align-pairs --num-samples 5000 --batch-size 50"
                "eval.total_eval --data magic_brush --num-samples 693 --batch-size 50 --split test"
                "eval.total_eval --data hq_edit_text_images --num-samples 2000 --batch-size 50"

                "eval.total_eval --data coco-triplets --num-samples 780 --batch-size 50 --split test"    
                "eval.total_eval --data polaris --num-samples 5000 --batch-size 50 --split test"
                "eval.total_eval --data hq_edit_text_2afc --num-samples 2000 --batch-size 50"
                    
                "eval.total_eval --data kadid-pairs --num-samples 5000 --batch-size 50 --mode naive"
                "eval.total_eval --data kadid-pairs --num-samples 5000 --batch-size 50 --mode clip-iqa"
                "eval.total_eval --data koniq-pairs --split test --num-samples 5000 --batch-size 50 --mode naive"
                "eval.total_eval --data koniq-pairs --num-samples 5000 --batch-size 50 --mode clip-iqa"
                "eval.total_eval --data pie-app-iqa --split test --num-samples 5000 --batch-size 50 --mode naive"
                "eval.total_eval --data pie-app-iqa --split test --num-samples 5000 --batch-size 50 --mode clip-iqa"
                "eval.total_eval --data agiqa3k-qual-pairs --num-samples 5000 --batch-size 50 --mode naive"
                "eval.total_eval --data agiqa3k-qual-pairs --num-samples 5000 --batch-size 50 --mode clip-iqa"
                "eval.total_eval --data pipal --split val --num-samples 3025 --batch-size 50 --mode naive"
                "eval.total_eval --data pipal --split val --num-samples 3025 --batch-size 50 --mode clip-iqa"
            
                "eval.total_eval --data sice-pairs-h --split test --num-samples 2151 --batch-size 50 --mode naive"
                "eval.total_eval --data sice-pairs-h --split test --num-samples 2151 --batch-size 50 --mode clip-iqa"
                "eval.total_eval --data koniq-brightness-pairs --split test --num-samples 5000 --batch-size 50 --mode naive"
                "eval.total_eval --data koniq-brightness-pairs --split test --num-samples 5000 --batch-size 50 --mode clip-iqa"
                "eval.total_eval --data koniq-colorfulness-pairs --split test --num-samples 5000 --batch-size 50 --mode naive"
                "eval.total_eval --data koniq-colorfulness-pairs --split test --num-samples 5000 --batch-size 50 --mode clip-iqa"
                "eval.total_eval --data koniq-contrast-pairs --split test --num-samples 5000 --batch-size 50 --mode naive"
                "eval.total_eval --data koniq-contrast-pairs --split test --num-samples 5000 --batch-size 50 --mode clip-iqa"
                "eval.total_eval --data koniq-sharpness-pairs --split test --num-samples 5000 --batch-size 50 --mode naive"
                "eval.total_eval --data koniq-sharpness-pairs --split test --num-samples 5000 --batch-size 50 --mode clip-iqa"

                "eval.total_eval --data h-imagenet-triplets --num-samples 5000 --batch-size 50" 
                "eval.total_eval --data cifar100coarse-triplets --num-samples 5000 --batch-size 50"
                
                "eval.total_eval --data roxford5k --num-samples 10000 --batch-size 50"
                "eval.total_eval --data rparis6k --num-samples 10000 --batch-size 50"
)

declare -a models=(#
    "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    "openai/clip-vit-large-patch14-336"
    "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    "hf-hub:timm/ViT-SO400M-14-SigLIP"
    "blip"
    "dreamsim:open_clip_vitb32"
    "dreamsim:ensemble"
    "ImageReward-v1.0"
    "hps"
    "pac-s"
    "liqe-mix" 
)

declare -a ckpts=(#
    ""
)
evaluate models[@] ckpts[@] tasks[@]

declare -a ckpts=(# 
                "--lora --ckptpath path/to/epoch_1.pt"
               )

declare -a models=(#
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
)

evaluate models[@] ckpts[@] tasks[@]

declare -a ckpts=(# 
                 "--lora --ckptpath path/to/epoch_1.pt"
                )

declare -a models=(#
  "openai/clip-vit-large-patch14-336"
)

evaluate models[@] ckpts[@] tasks[@]