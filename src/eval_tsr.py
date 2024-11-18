import argparse
import json


def compute_tsr(src_list, hypo_list, tgt_lang):
    sum = 0
    total = 0
    for src, hypo in zip(src_list, hypo_list):
        total += len(src['hints'])
        for hint in src['hints']:
            if hint[tgt_lang] in hypo:
                #print(hint)
                sum += 1
    print(sum)
    print(total)
    return sum/total


def main():
    parser = argparse.ArgumentParser(description="Script with conditional parameters")
    parser.add_argument('--src_file', type=str)
    parser.add_argument('--hypo_file', type=str)
    parser.add_argument('--lang_pair', type=str)
    parser.add_argument('--record_file', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    src_list = [json.loads(line) for line in open(args.src_file, 'r', encoding='utf-8')]
    hypo_list = [line.strip() for line in open(args.hypo_file, "r", encoding="utf-8").readlines()]
    assert len(src_list) == len(hypo_list)
    src_lang, tgt_lang = args.lang_pair.split("-")
    TSR = compute_tsr(src_list, hypo_list, tgt_lang)
    with open(args.record_file, "a", encoding="utf-8") as file:
        file.write(args.model + "\t" + args.lang_pair + "\t" + "{:.2%}".format(TSR) + "\n")


if __name__ == "__main__":
    main()