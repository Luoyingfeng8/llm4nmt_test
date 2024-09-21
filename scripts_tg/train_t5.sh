#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

config_file=./configs/accelerate_config_bf16.yaml

model_name=mt5-large
model_dir=/mnt/luoyingfeng/model_card/$model_name
resume_from_checkpoint=/mnt/luoyingfeng/llm4nmt/exps/mt5-large/10M_s1/checkpoint-12000

language_pairs=de-en,en-de,cs-en,en-cs,ru-en,en-ru,zh-en,en-zh
# language_pairs=en-de
mmt_data_path=/mnt/luoyingfeng/llm4nmt/data/wmt23-sample10M
trans_task="general_trans"
tag=10M_s1

output_dir=/mnt/luoyingfeng/llm4nmt/exps/$model_name/$tag
mkdir -p $output_dir
cp $0 $output_dir

accelerate launch --config_file $config_file $ROOT_DIR/src/run_translation.py \
    --model_name_or_path $model_dir \
    --resume_from_checkpoint ${resume_from_checkpoint:-""} \
    --mmt_data_path $mmt_data_path \
    --trans_task $trans_task \
    --test_dataname wmt23 \
    --language_pairs $language_pairs \
    --use_fast_tokenizer \
    --do_train \
    --do_eval \
    --do_predict \
    --predict_with_generate \
    --num_beams 5 \
    --output_dir $output_dir \
    --cache_dir ./cache \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --load_best_model_at_end  \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --max_source_length 256 \
    --max_target_length 256 \
    --num_train_epochs 3 \
    --patience 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_strategy steps \
    --eval_steps 1000  \
    --save_steps 1000  \
    --logging_steps 10 \
    --save_total_limit  5 \
    --bf16 True \
    --seed 42 \
    --report_to "tensorboard" \
    --overwrite_output_dir True \
    | tee $output_dir/train.log

 bash ./eval_multi.sh  $output_dir/decode_result  