#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

config_file=./configs/accelerate_config_bf16.yaml

model_name=mt5-large
model_dir=$ROOT_DIR/model_card/$model_name
# resume_from_checkpoint=/mnt/luoyingfeng/llm4nmt/exps/mt5-large/10M_s1/checkpoint-12000

## train
language_pairs=nb-en,zh-en,dz-en,wa-en,ha-en,or-en,nl-en,hu-en,sv-en,ug-en,ko-en,ur-en,mn-en,yi-en,sk-en,mr-en,ja-en,is-en,tt-en,he-en,es-en,xh-en,kk-en,fi-en,gl-en,pl-en,ps-en,lt-en,id-en,vi-en,bs-en,eu-en,ku-en,sr-en,ca-en,tr-en,li-en,ml-en,hr-en,el-en,ig-en,ne-en,ru-en,mg-en,sh-en,af-en,mk-en,fy-en,be-en,fr-en,de-en,my-en,eo-en,lv-en,rw-en,bg-en,tg-en,zu-en,bn-en,mt-en,nn-en,ga-en,no-en,et-en,gu-en,ta-en,cy-en,ms-en,kn-en,as-en,br-en,yo-en,se-en,uz-en,gd-en,az-en,hy-en,uk-en,sq-en,te-en,da-en,si-en,am-en,tk-en,fa-en,cs-en,ka-en,ro-en,an-en,ky-en,oc-en,sl-en,pa-en,pt-en,th-en,it-en,ar-en,km-en,hi-en
## test
# language_pairs=nb-en,zh-en,dz-en,wa-en,ha-en,or-en,nl-en,hu-en,sv-en,ug-en,ko-en,ur-en,mn-en,yi-en,sk-en,mr-en,ja-en,is-en,tt-en,he-en,es-en,xh-en,kk-en,fi-en,gl-en,pl-en,ps-en,lt-en,id-en,vi-en,bs-en,eu-en,ku-en,sr-en,ca-en,tr-en,li-en,ml-en,hr-en,el-en,ig-en,ne-en,ru-en,mg-en,sh-en,af-en,mk-en,fy-en,be-en,fr-en,de-en,my-en,eo-en,lv-en,rw-en,bg-en,tg-en,zu-en,bn-en,mt-en,nn-en,ga-en,no-en,et-en,gu-en,ta-en,cy-en,ms-en,kn-en,as-en,br-en,yo-en,se-en,uz-en,gd-en,az-en,hy-en,uk-en,sq-en,te-en,da-en,si-en,am-en,tk-en,fa-en,cs-en,ka-en,ro-en,an-en,ky-en,oc-en,sl-en,pa-en,pt-en,th-en,it-en,ar-en,km-en,hi-en

mmt_data_path=$ROOT_DIR/data/opus-flores
trans_task="general_trans"
tag=opus

output_dir=$ROOT_DIR/exps/$model_name/$tag
mkdir -p $output_dir
cp $0 $output_dir

accelerate launch --config_file $config_file $ROOT_DIR/src/run_translation.py \
    --model_name_or_path $model_dir \
    --resume_from_checkpoint ${resume_from_checkpoint:-""} \
    --mmt_data_path $mmt_data_path \
    --trans_task $trans_task \
    --language_pairs $language_pairs \
    --use_fast_tokenizer \
    --do_train \
    --do_eval \
    --do_predict \
    --predict_with_generate \
    --num_beams 4 \
    --output_dir $output_dir \
    --cache_dir ./cache \
    --learning_rate 8e-5 \
    --weight_decay 0.01 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --load_best_model_at_end  \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --max_source_length 256 \
    --max_target_length 256 \
    --num_train_epochs 1 \
    --patience 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_strategy steps \
    --eval_steps 5000  \
    --save_steps 1000  \
    --logging_steps 10 \
    --save_total_limit  5 \
    --bf16 True \
    --seed 42 \
    --report_to "tensorboard" \
    --overwrite_output_dir True \
    | tee $output_dir/train.log

#  bash ./eval_opus.sh  $output_dir/decode_result  