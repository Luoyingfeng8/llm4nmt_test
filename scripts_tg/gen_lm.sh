#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

eval_file=./eval_result.txt
config_file=./configs/accelerate_config.yaml

# language_pairs=de-en,en-de,cs-en,en-cs,ru-en,en-ru,zh-en,en-zh
language_pairs=zh-en
# trans_task="doc_trans,domain_medical,domain_law,domain_academic,domain_social_media,domain_xml,term_con_trans,ape,context_aware_trans,context_learning_trans"
trans_task=general_trans
# ref_file=/media/luoyf/data_card/iwslt14-deen/test.$tgt
mmt_data_path=$ROOT_DIR/data/final

eval_models=(
/mnt/luoyingfeng/llm4nmt/exps/Meta-Llama-3-8B/sft_x2en/checkpoint-680
)

for eval_model in ${eval_models[@]}; do
    model_dir=$eval_model
    output_dir=$model_dir
    cp $0 $output_dir
    
    # CUDA_VISIBLE_DEVICES=5 python  $ROOT_DIR/src/run_llmmt.py \
    accelerate launch --config_file $config_file  $ROOT_DIR/src/run_llmmt.py \
        --model_name_or_path $model_dir \
        --mmt_data_path $mmt_data_path \
        --use_fast_tokenizer \
        --do_predict \
        --predict_with_generate \
        --language_pairs $language_pairs \
        --trans_task $trans_task \
        --low_cpu_mem_usage \
        --per_device_eval_batch_size 2 \
        --output_dir  $output_dir \
        --max_source_length 512 \
        --seed 42 \
        --overwrite_output_dir \
        --num_beams 5 \
        --max_new_tokens 512 \
        --overwrite_cache True \
        --torch_dtype "auto"


    bash eval_multi_new.sh $output_dir/decode_result

done

