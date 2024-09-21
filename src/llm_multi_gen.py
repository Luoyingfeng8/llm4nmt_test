#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import logging
from typing import Tuple
import tqdm
import time

import torch
from accelerate import PartialState, Accelerator
from accelerate.utils import set_seed, gather_object

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)
from functools import reduce
from utils import utils
import datetime

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))

def make_model_prompt(model_name_or_path, test_dataset, src_fullname, tgt_fullname, tokenizer):
    if "ALMA" in model_name_or_path or "gemma-2" in model_name_or_path:
        prefix = "Translate this from {src_fullname} to {tgt_fullname}:\n{src_fullname}: {src}\n{tgt_fullname}:"
        res = []
        for line in test_dataset:
            prompt = prefix.format(src_fullname=src_fullname, tgt_fullname=tgt_fullname, src=line)
            res.append(prompt)
    elif "Tower" in model_name_or_path:
        prefix = "Translate the following text from {src_fullname} into {tgt_fullname}.\n{src_fullname}: {src}\n{tgt_fullname}:"
        res = []
        for line in test_dataset:
            prompt = prefix.format(src_fullname=src_fullname, tgt_fullname=tgt_fullname, src=line)
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            res.append(text)
    elif "Meta-Llama-3.1" in model_name_or_path or "Meta-Llama-3-8B" in model_name_or_path:
        prefix = "Translate the following text from {src_fullname} into {tgt_fullname}. Do not provide any explanations or text apart from the translation.\n{src}"
        res = []
        for line in test_dataset:
            prompt = prefix.format(src_fullname=src_fullname, tgt_fullname=tgt_fullname, src=line)
            messages = [
                {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            res.append(text)
    elif "nllb" in model_name_or_path:
        res = test_dataset
    else:
        print("Not support this model")
        exit()
    
    return res

def clean_pred(pred, remove_special_tokens=[]):
    ## remove special tokens
    for s in remove_special_tokens:
        pred = pred.replace(s, "")
    ## last step: check
    pred = "#" if utils.is_whitespace(pred) else pred
    return pred

def get_special_tokens(tokenizer):
    remove_special_tokens = ["<unk>", "</s>", "<pad>", "\n"]
    if getattr(tokenizer, "pad_token", None):
        remove_special_tokens.append(tokenizer.pad_token)
    if getattr(tokenizer, "eos_token", None):
        remove_special_tokens.append(tokenizer.eos_token)
    if getattr(tokenizer, "bos_token", None):
        remove_special_tokens.append(tokenizer.bos_token)
    if getattr(tokenizer, "unk_token", None):
        remove_special_tokens.append(tokenizer.unk_token)
    return remove_special_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--hypo_file", type=str, default="")
    parser.add_argument("--lang_pair", type=str, default='de-en')
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 1.0 has no effect, lower tend toward greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2" )
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--topp", type=float, default=1.0)
    parser.add_argument("--num_batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    args = parser.parse_args()
    set_seed(args.seed)
    accelerator = Accelerator()

    src_lang, tgt_lang = args.lang_pair.split("-")
    src_fullname = utils.LANG_TABLE[src_lang]
    tgt_fullname = utils.LANG_TABLE[tgt_lang]
    # for nllb model
    langcodes = {"en": "eng_Latn", "de":"deu_Latn", "cs":"ces_Latn", "ru":"rus_Cyrl", "zh":"zho_Hans"}

    # Initialize the distributed state.        
    if "nllb" in args.model_name_or_path:
        # https://github.com/facebookresearch/flores/tree/main/flores200
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_auth_token=True, src_lang=langcodes[src_lang])
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, add_special_tokens=False, padding_side="left", trust_remote_code=True)
    remove_special_tokens = get_special_tokens(tokenizer)

    if "Llama-2" in args.model_name_or_path:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    if "Llama-3" in args.model_name_or_path:
        tokenizer.pad_token_id = 128002

    torch_dtype = 'auto'
    # torch_dtype = torch.bfloat16
    # torch_dtype = torch.float16
    if "nllb" in args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path, 
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map={"": accelerator.process_index},
        )
        forced_bos_token_id = tokenizer.lang_code_to_id[langcodes[tgt_lang]]
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map={"": accelerator.process_index},
        )
        forced_bos_token_id = None

    test_dataset = open(args.test_file, encoding='utf8').read().strip().split("\n")
    

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start = time.time()

    test_dataset = make_model_prompt(args.model_name_or_path, test_dataset, src_fullname, tgt_fullname, tokenizer)

    # batch, left pad (for inference), and tokenize
    def make_batch(prompts, batch_size=4):
        batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
        batches_tok = []
        for prompt_batch in batches:
            input_ids = tokenizer(
                    prompt_batch, 
                    return_tensors="pt", 
                    padding='longest', 
                    truncation=False, 
                    pad_to_multiple_of=8).to("cuda") 
            batches_tok.append(input_ids)
                
        return batches_tok

    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(test_dataset) as prompts:
        results = dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches = make_batch(prompts, batch_size=args.num_batch)
        prompt_batches = tqdm.tqdm(prompt_batches, total=len(prompt_batches), disable=not accelerator.is_local_main_process)
        for prompts_tokenized in prompt_batches:
            outputs_tokenized = model.generate(
                **prompts_tokenized,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                do_sample=False,
                num_return_sequences=args.num_return_sequences,
                forced_bos_token_id=forced_bos_token_id
            )

            
            if "nllb" in args.model_name_or_path:
                num_tokens = sum([ len(t) for t in outputs_tokenized ])
                outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
            # remove prompt from gen. tokens
            else:
                outputs_tokenized = [ tok_out[len(tok_in):] 
                    for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

                # count and decode gen. tokens 
                num_tokens = sum([ len(t) for t in outputs_tokenized ])
                outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
            
            # print("\n\n".join(outputs))
            # store in results{} to be gathered by accelerate
            outputs = list(map(lambda x: clean_pred(x, remove_special_tokens=remove_special_tokens),  outputs))
            # print(outputs)
            results["outputs"].extend(outputs)
            results["num_tokens"] += num_tokens
    results = [ results ]

    # collect results from all the GPUs
    results_gathered = gather_object(results)
    
    if accelerator.is_main_process:
        timediff = time.time() - start
        num_tokens = sum([r["num_tokens"] for r in results_gathered ])
        preds = list(reduce(lambda x,y: x+y["outputs"], results_gathered, []))
        print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")
        with open(args.hypo_file, mode='w') as fout:
            fout.write("\n".join(preds) + '\n')
                
if __name__ == "__main__":
    main()