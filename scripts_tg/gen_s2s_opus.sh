#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

config_file=./configs/accelerate_config_8gpu.yaml

language_pairs=bn-en,en-it,en-ps,en-kk,en-sv,en-ky,en-wa,en-lv,en-ka,en-nn,en-si,en-or,az-en,en-ha,de-en,en-th,en-mt,en-he,en-oc,en-ur,en-ku,en-xh,en-tr,en-gd,en-lt,cs-en,en-yi,en-eu,en-kn,en-nb,en-es,en-te,en-hi,en-zh,el-en,en-id,en-fy,br-en,en-uk,en-gl,en-mk,en-sl,en-pl,as-en,en-pt,en-tg,en-ja,en-et,en-pa,en-my,ar-en,en-sk,en-fr,en-mr,en-mg,en-hu,bg-en,af-en,en-hr,en-fa,en-se,da-en,en-ga,en-ig,en-li,en-no,en-uz,en-rw,en-ml,en-gu,en-fi,be-en,en-ta,en-tk,en-ug,en-ne,am-en,en-ms,en-is,en-sr,en-sq,cy-en,en-tt,en-ru,en-vi,en-eo,en-km,bs-en,ca-en,en-sh,en-ro,en-zu,en-nl,en-ko
# language_pairs=de-en,cs-en,ru-en,zh-en

mmt_data_path=$ROOT_DIR/data/opus
trans_task="general_trans"

eval_models=(
/mnt/luoyingfeng/llm4nmt/exps/my_experiment/cross_atten_L8_D1024_m2m_s1_gfuse/checkpoint-33000
)


for eval_model in ${eval_models[@]}; do
    # model_dir=/mnt/luoyingfeng/llm4nmt/exps/wmt22-multi/$model_tag/$eval_model
    model_dir=$eval_model
    output_dir=$model_dir
    cp $0 $output_dir

    batch_size=4

    accelerate launch --config_file $config_file $ROOT_DIR/src/run_translation.py \
        --model_name_or_path $model_dir \
        --mmt_data_path $mmt_data_path \
        --trans_task $trans_task \
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

    # bash ./eval_multi_new.sh  $output_dir/decode_result 

done