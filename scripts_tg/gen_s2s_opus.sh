#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

config_file=./configs/accelerate_config_8gpu.yaml

language_pairs=nb-en,zh-en,dz-en,wa-en,ha-en,or-en,nl-en,hu-en,sv-en,ug-en,ko-en,ur-en,mn-en,yi-en,sk-en,mr-en,ja-en,is-en,tt-en,he-en,es-en,xh-en,kk-en,fi-en,gl-en,pl-en,ps-en,lt-en,id-en,vi-en,bs-en,eu-en,ku-en,sr-en,ca-en,tr-en,li-en,ml-en,hr-en,el-en,ig-en,ne-en,ru-en,mg-en,sh-en,af-en,mk-en,fy-en,be-en,fr-en,de-en,my-en,eo-en,lv-en,rw-en,bg-en,tg-en,zu-en,bn-en,mt-en,nn-en,ga-en,no-en,et-en,gu-en,ta-en,cy-en,ms-en,kn-en,as-en,br-en,yo-en,se-en,uz-en,gd-en,az-en,hy-en,uk-en,sq-en,te-en,da-en,si-en,am-en,tk-en,fa-en,cs-en,ka-en,ro-en,an-en,ky-en,oc-en,sl-en,pa-en,pt-en,th-en,it-en,ar-en,km-en,hi-en
# language_pairs=de-en,cs-en,ru-en,zh-en

mmt_data_path=$ROOT_DIR/data/opus-flores
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