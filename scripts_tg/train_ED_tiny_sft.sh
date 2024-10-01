#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

model_name=Meta-Llama-3-8B

config_file=./configs/deepspeed_train_config_bf16_8gpu.yaml

# model_dir=/mnt/luoyingfeng/model_card/$model_name
model_dir=/mnt/luoyingfeng/llm4nmt/exps/Meta-Llama-3-8B/stack_L8_D1024_m2m_s1_gfuse/checkpoint-52000
# resume_from_checkpoint=/mnt/luoyingfeng/llm4nmt/exps/wmt22-multi/mt5-xl
run_mode="continue"

model_method="TinyCrossAttLW"
# encoder_method="bidirectional"
encoder_method="stack"
encoder_layer_num=8
decoder_layer_num=4
decoder_hidden_size=1024
decoder_intermediate_size=2752
decoder_num_attention_heads=16
decoder_num_key_value_heads=16

decoder_param_method="freeze"  # no use
tag=stack_L8_D1024_m2m_s2

# language_pairs=de-en,cs-en,ru-en,zh-en
language_pairs=de-en,en-de,cs-en,en-cs,ru-en,en-ru,zh-en,en-zh
mmt_data_path=$ROOT_DIR/data/v8.28
trans_task="general_trans,doc_trans,domain_medical,domain_law,domain_finance,domain_computer,domain_literature,domain_social_media,term_con_trans,ape,context_aware_trans,context_learning_trans"


# resume_from_checkpoint=/mnt/luoyingfeng/llm4nmt/exps/Meta-Llama-3-8B/stack_L8_D1024_m2m_s2/checkpoint-1269

output_dir=$ROOT_DIR/exps/$model_name/$tag
mkdir -p $output_dir
cp $0 $output_dir

accelerate launch --config_file $config_file $ROOT_DIR/src/run_translation.py \
    --model_name_or_path $model_dir \
    --resume_from_checkpoint ${resume_from_checkpoint:-""} \
    --encoder_layer_num ${encoder_layer_num} \
    --decoder_layer_num $decoder_layer_num \
    --decoder_hidden_size $decoder_hidden_size \
    --decoder_intermediate_size $decoder_intermediate_size \
    --decoder_num_attention_heads $decoder_num_attention_heads \
    --decoder_num_key_value_heads $decoder_num_key_value_heads \
    --model_method ${model_method:-"norm"} \
    --encoder_method ${encoder_method} \
    --run_mode ${run_mode:-""} \
    --decoder_param_method ${decoder_param_method:-"share"} \
    --mmt_data_path $mmt_data_path \
    --trans_task $trans_task \
    --test_dataname wmt23 \
    --language_pairs $language_pairs \
    --use_fast_tokenizer \
    --do_train \
    --do_eval \
    --do_predict \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --load_best_model_at_end  \
    --cache_dir ./cache \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --max_source_length 512 \
    --max_target_length 512 \
    --output_dir  $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --num_beams 5 \
    --max_new_tokens 512 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_strategy steps \
    --eval_steps  0.1 \
    --save_steps 0.1 \
    --logging_steps  0.01 \
    --save_total_limit  5 \
    --fp16 \
    --seed 42 \
    --report_to "tensorboard" \
    --overwrite_output_dir True \
   | tee $output_dir/train.log
    

 bash ./eval_multi_new.sh  $output_dir/decode_result 