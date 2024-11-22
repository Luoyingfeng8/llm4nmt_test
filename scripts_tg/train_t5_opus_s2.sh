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
# language_pairs=en-nb,en-zh,dz-en,en-wa,en-ha,en-or,en-nl,en-hu,en-sv,en-ug,en-ko,en-ur,en-mn,en-yi,en-sk,en-mr,en-ja,en-is,en-tt,en-he,en-es,en-xh,en-kk,en-fi,en-gl,en-pl,en-ps,en-lt,en-id,en-vi,bs-en,en-eu,en-ku,en-sr,ca-en,en-tr,en-li,en-ml,en-hr,el-en,en-ig,en-ne,en-ru,en-mg,en-sh,af-en,en-mk,en-fy,be-en,en-fr,de-en,en-my,en-eo,en-lv,en-rw,bg-en,en-tg,en-zu,bn-en,en-mt,en-nn,en-ga,en-no,en-et,en-gu,en-ta,cy-en,en-ms,en-kn,as-en,br-en,en-yo,en-se,en-uz,en-gd,az-en,en-hy,en-uk,en-sq,en-te,da-en,en-si,am-en,en-tk,en-fa,cs-en,en-ka,en-ro,an-en,en-ky,en-oc,en-sl,en-pa,en-pt,en-th,en-it,ar-en,en-km,en-hi

## test
language_pairs=bn-en,en-it,en-ps,en-kk,en-sv,en-ky,en-wa,en-lv,en-ka,en-nn,en-si,en-or,az-en,en-ha,de-en,en-th,en-mt,en-he,en-oc,en-ur,en-ku,en-xh,en-tr,en-gd,en-lt,cs-en,en-yi,en-eu,en-kn,en-nb,en-es,en-te,en-hi,en-zh,el-en,en-id,en-fy,br-en,en-uk,en-gl,en-mk,en-sl,en-pl,as-en,en-pt,en-tg,en-ja,en-et,en-pa,en-my,ar-en,en-sk,en-fr,en-mr,en-mg,en-hu,bg-en,af-en,en-hr,en-fa,en-se,da-en,en-ga,en-ig,en-li,en-no,en-uz,en-rw,en-ml,en-gu,en-fi,be-en,en-ta,en-tk,en-ug,en-ne,am-en,en-ms,en-is,en-sr,en-sq,cy-en,en-tt,en-ru,en-vi,en-eo,en-km,bs-en,ca-en,en-sh,en-ro,en-zu,en-nl,en-ko

mmt_data_path=$ROOT_DIR/data/opus-s2
trans_task="general_trans"
tag=opus-s2

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
    --num_train_epochs 2 \
    --patience 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 6 \
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

#  bash ./eval_opus.sh  $output_dir/decode_result  