CACHE_DIR='all_models/'
DATA_DIR='/uni_data'
MODEL_DIR='all_models/'

export HF_HOME=$CACHE_DIR

declare -a tasks=(#

                "eval.total_eval --data night --split test --num-samples 1824 --batch-size 1"
                "eval.total_eval --data bapps --split val --num-samples 5000 --batch-size 1"
                "eval.total_eval --data pie-app-2afc --split test --num-samples 3314 --batch-size 1"
                    
                "eval.total_eval --data imagereward1k-pairs --split validation --num-samples 412 --batch-size 1"
                "eval.total_eval --data hpdv2-pairs --num-samples 5000 --batch-size 1 --split test"
                "eval.total_eval --data agiqa3k-align-pairs --num-samples 5000 --batch-size 1"
                "eval.total_eval --data magic_brush --num-samples 693 --batch-size 1 --split test"
                "eval.total_eval --data hq_edit_text_images --num-samples 2000 --batch-size 1"

                "eval.total_eval --data coco-triplets --num-samples 780 --batch-size 1 --split test"    
                "eval.total_eval --data polaris --num-samples 5000 --batch-size 1 --split test"
                "eval.total_eval --data hq_edit_text_2afc --num-samples 2000 --batch-size 1"
                    
                "eval.total_eval --data kadid-pairs --num-samples 5000 --batch-size 1 --mode naive"

                "eval.total_eval --data koniq-pairs --split test --num-samples 5000 --batch-size 1 --mode naive"

                "eval.total_eval --data pie-app-iqa --split test --num-samples 5000 --batch-size 1 --mode naive"

                "eval.total_eval --data agiqa3k-qual-pairs --num-samples 5000 --batch-size 1 --mode naive"

                "eval.total_eval --data pipal --split val --num-samples 3025 --batch-size 1 --mode naive"
            
                "eval.total_eval --data sice-pairs-h --split test --num-samples 2151 --batch-size 1 --mode naive"

                "eval.total_eval --data koniq-brightness-pairs --split test --num-samples 5000 --batch-size 1 --mode naive"

                "eval.total_eval --data koniq-colorfulness-pairs --split test --num-samples 5000 --batch-size 1 --mode naive"

                "eval.total_eval --data koniq-contrast-pairs --split test --num-samples 5000 --batch-size 1 --mode naive"

                "eval.total_eval --data koniq-sharpness-pairs --split test --num-samples 5000 --batch-size 1 --mode naive"

                "eval.total_eval  --data h-imagenet-triplets --num-samples 5000 --batch-size 1" 
                "eval.total_eval --data cifar100coarse-triplets --num-samples 5000 --batch-size 1"
)
declare -a ckpts=(#
                 "lmms-lab/llava-next-interleave-qwen-0.5b"
                 "lmms-lab/llava-next-interleave-qwen-7b"
                 "TIGER-Lab/Mantis-8B-Idefics2"
                  )
                  
for ckpt in "${ckpts[@]}"
    do
    for task in "${tasks[@]}"
    do
        $SCRATCH/pytorch-example/python -m $task \
            --cache_dir ./ \
            --data-path $DATA_DIR \
            --model-path $ckpt \
            \
            --temperature 0 \
            --num-workers 10 \
            \
            \
            --seed 0 
    done
done