# !/bin/bash
set -eux

ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

comet_model=$ROOT_DIR/model_card/wmt22-comet-da/checkpoints/model.ckpt 
xcome_model=$ROOT_DIR/model_card/XCOMET-XXL/checkpoints/model.ckpt

decode_dir=${1:-"/mnt/luoyingfeng/llm4nmt/exps/Meta-Llama-3-8B/sft_2epoch/decode_result"}

# hypo_files=(
# $decode_dir/test-en-zh-general_trans-wmt23
# $decode_dir/test-en-zh-ape
# $decode_dir/test-en-zh-doc_trans
# $decode_dir/test-en-zh-domain_colloquial
# $decode_dir/test-en-zh-domain_literature
# $decode_dir/test-en-zh-domain_medical
# $decode_dir/test-en-zh-term_con_trans
# )

hypo_files=(
$decode_dir/test-en-zh-doc_trans
$decode_dir/test-en-zh-domain_colloquial
$decode_dir/test-en-zh-domain_literature
$decode_dir/test-en-zh-domain_medical
)


# eval_models=(ALMA-13B ALMA-7B aya-23-8B nllb-200-3.3B TowerInstruct-7B-v0.2 TowerInstruct-13B-v0.2 Meta-Llama-3-8B-Instruct)
# eval_models=(aya-23-8B TowerInstruct-7B-v0.2 TowerInstruct-13B-v0.2 Meta-Llama-3-8B-Instruct ALMA-7B ALMA-13B nllb-200-3.3B)
eval_models=(A1)
for eval_model in ${eval_models[@]}; do

            
    src_file_strs=""
    ref_file_strs=""
    hypo_file_strs=""
    lang_pair_strs=""

    for hypo_file in ${hypo_files[@]}; do 

        # 提取文件名部分
        filename=$(basename "$hypo_file")

        # 移除前缀 'test-' 并使用 '-' 分割
        filename=${filename#test-}
        filename=${filename%-new}
        IFS='-' read -r src_lang tgt_lang task_type <<< "$filename"
        
        if [ "$src_lang" != "en" ]; then
            first_lang="$src_lang"
        else
            first_lang="$tgt_lang"
        fi
        
        lp=${src_lang}-${tgt_lang}
        lp2=${src_lang}2${tgt_lang}


        src_file=$ROOT_DIR/data/v12.12_txt/${first_lang}-en/test.$lp.$task_type.$src_lang.txt
        ref_file=$ROOT_DIR/data/v12.12_txt/${first_lang}-en/test.$lp.$task_type.$tgt_lang.txt	
        
        src_file_strs=${src_file_strs:+$src_file_strs,}$src_file
        ref_file_strs=${ref_file_strs:+$ref_file_strs,}$ref_file
        hypo_file_strs=${hypo_file_strs:+$hypo_file_strs,}$hypo_file
        lang_pair_strs=${lang_pair_strs:+$lang_pair_strs,}$lp2
            
    done


    python $ROOT_DIR/llm4nmt/src/mt_scoring.py \
        --metric "bleu,comet_22"  \
        --comet_22_path $comet_model \
        --xcomet_xxl_path $xcome_model \
        --lang_pair $lang_pair_strs \
        --src_file $src_file_strs \
        --ref_file $ref_file_strs \
        --hypo_file $hypo_file_strs \
        --record_file "ComMT_result.xlsx" \
        --write_key "suffix" \
        --gpu 0
done



