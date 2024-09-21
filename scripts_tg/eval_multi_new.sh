# !/bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1



decode_dir=${1:-"/mnt/luoyingfeng/llm4nmt/exps/TowerInstruct-13B-v0.2/wmt23"}
wmt_test_year=wmt23
# suffix=-general_trans-$wmt_test_year

eval_file=./eval_result.txt
rm -f ./temp.txt

comet_model=$ROOT_DIR/model_card/wmt22-comet-da/checkpoints/model.ckpt 
xcome_model=$ROOT_DIR/model_card/XCOMET-XXL/checkpoints/model.ckpt

src_file_strs=""
ref_file_strs=""
hypo_file_strs=""
lang_pair_strs=""

for l in de cs ru zh; do
    for src in $l en; do
    # for src in en; doeval_file
        if [ $src = "en" ]; then
            tgt=$l
        else 
            tgt=en
        fi

        if [ $src = "cs" ] && [ $wmt_test_year = "wmt23" ]; then
            continue
        fi
        
        lang_pair=${src}-$tgt
        lp=${src}2${tgt}
        hypo_file=$decode_dir/test-$lang_pair${suffix:-""}
        src_file=$ROOT_DIR/data/wmt-test-data/${wmt_test_year}/${wmt_test_year}-${l}en/test.$lp.$src
        ref_file=$ROOT_DIR/data/wmt-test-data/${wmt_test_year}/${wmt_test_year}-${l}en/test.$lp.$tgt	
        
        src_file_strs=${src_file_strs:+$src_file_strs,}$src_file
        ref_file_strs=${ref_file_strs:+$ref_file_strs,}$ref_file
        hypo_file_strs=${hypo_file_strs:+$hypo_file_strs,}$hypo_file
        lang_pair_strs=${lang_pair_strs:+$lang_pair_strs,}$lp
        
    done
done

python $ROOT_DIR/src/mt_scoring.py \
    --metric "bleu,comet_22,xcomet_xxl"  \
    --comet_22_path $comet_model \
    --xcomet_xxl_path $xcome_model \
    --lang_pair $lang_pair_strs \
    --src_file $src_file_strs \
    --ref_file $ref_file_strs \
    --hypo_file $hypo_file_strs \
    --record_file "result.xlsx"


