# !/bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

decode_dir=${1:-"/mnt/luoyingfeng/llm4nmt/exps/Meta-Llama-3-8B/stack_E8_L8_D1024_opus100_s1/checkpoint-56000/decode_result"}


for filename in `ls $decode_dir`; do
    src=$(echo $filename | cut -d'-' -f2)
    tgt=$(echo $filename | cut -d'-' -f3)
    lang_pair=$src-$tgt
    
    if [ $src == "en" ]; then
        first_lang=$tgt
    else
        first_lang=$src
    fi

    hypo_file=$decode_dir/$filename
    ref_file=$ROOT_DIR/data/opus-flores/$first_lang-en/test.$lang_pair.general_trans.$tgt.txt
    

    cur_time=`date +"%Y-%m-%d %H:%M:%S"`
    echo "=============$cur_time===================" >> ./temp.txt
    echo $lang_pair >> ./temp.txt
    echo $hypo_file >> ./temp.txt
    
    sacre_bleu=`sacrebleu -w 2 -b $ref_file -i $hypo_file -l $lang_pair`
       
    echo "sacre_bleu: $sacre_bleu" >> ./temp.txt

done