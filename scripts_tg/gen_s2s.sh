#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

config_file=./configs/accelerate_config_8gpu.yaml

language_pairs=de-en,en-de,cs-en,en-cs,ru-en,en-ru,zh-en,en-zh
# language_pairs=de-en,cs-en,ru-en,zh-en
# language_pairs=de-en
# language_pairs=en-de

mmt_data_path=$$ROOT_DIR/data/v8.28
trans_task="general_trans"

eval_models=(
/mnt/luoyingfeng/llm4nmt/exps/my_experiment/cross_atten_L8_D1024_m2m_s1_gfuse/checkpoint-33000
)


for eval_model in ${eval_models[@]}; do
    # model_dir=/mnt/luoyingfeng/llm4nmt/exps/wmt22-multi/$model_tag/$eval_model
    model_dir=$eval_model
    model_method="TinyCrossAttLW"
    output_dir=$model_dir
    cp $0 $output_dir

    batch_size=8

    accelerate launch --config_file $config_file $ROOT_DIR/src/run_translation.py \
        --model_name_or_path $model_dir \
        --model_method $model_method \
        --mmt_data_path $mmt_data_path \
        --trans_task $trans_task \
        --test_dataname wmt23 \
        --language_pairs $language_pairs \
        --use_fast_tokenizer \
        --do_predict \
        --predict_with_generate \
        --num_beams 5 \
        --max_new_tokens 256 \
        --cache_dir ./cache \
        --dataloader_num_workers 4 \
        --max_source_length 512 \
        --max_target_length 512 \
        --output_dir  $output_dir \
        --per_device_eval_batch_size $batch_size \
        --fp16 \
        --seed 42 \

    bash ./eval_multi_new.sh  $output_dir/decode_result 

done