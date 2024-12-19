#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

## only data parallel
config_file=./configs/accelerate_config_8gpu.yaml

## model
model_name=Meta-Llama-3-8B
model_dir=$ROOT_DIR/model_card/$model_name
# resume_from_checkpoint=/mnt/luoyingfeng/llm4nmt/exps/Meta-Llama-3-8B/stack_L8_D1024_m2m_s1/checkpoint-30000
run_mode="init"

model_method="TinyCrossAttLW"
encoder_method="stack"
# encoder_method="project"
encoder_layer_num=8
decoder_layer_num=8
decoder_hidden_size=1024
decoder_intermediate_size=2752
decoder_num_attention_heads=16
decoder_num_key_value_heads=16

decoder_param_method="freeze"
tag=${encoder_method}_E${encoder_layer_num}_D${decoder_layer_num}_d${decoder_hidden_size}_m2m_s1

## data
language_pairs=en-zh
# language_pairs=de-en,en-de,cs-en,en-cs,ru-en,en-ru,zh-en,en-zh
# mmt_data_path=$ROOT_DIR/data/wmt23-sample/wmt23-sample10M
mmt_data_path=$ROOT_DIR/data/wmt23-zhen
# mmt_data_path=/mnt/luoyingfeng/llm4nmt/data/wmt23-sample5M
trans_task="general_trans"
predict_task="general_trans,doc_trans,domain_medical,domain_law,domain_it,domain_literature,domain_colloquial"

epoch=2
batch_size=32 
gradient_accumulation=6

## save
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
    --encoder_method $encoder_method \
    --model_method ${model_method:-"norm"} \
    --run_mode ${run_mode:-""} \
    --decoder_param_method ${decoder_param_method:-"share"} \
    --mmt_data_path $mmt_data_path \
    --trans_task $trans_task \
    --predict_task $predict_task \
    --test_dataname wmt23 \
    --language_pairs $language_pairs \
    --use_fast_tokenizer \
    --do_eval \
    --do_train \
    --do_predict \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --load_best_model_at_end  \
    --cache_dir ./cache \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --max_source_length 256 \
    --max_target_length 256 \
    --output_dir  $output_dir \
    --num_train_epochs $epoch \
    --patience 3 \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $gradient_accumulation \
    --predict_with_generate \
    --num_beams 5 \
    --max_new_tokens 512 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_strategy steps \
    --eval_steps  1000 \
    --save_steps 1000 \
    --logging_steps  50 \
    --save_total_limit  5 \
    --fp16 \
    --seed 42 \
    --report_to "tensorboard" \
    --overwrite_output_dir True \
   | tee $output_dir/train.log
    

#  bash ./eval_multi_new_commt.sh  $output_dir/decode_result 