# !/bin/bash
set -eux
cd ..
mkdir -p model_card && mv Meta-Llama-3-8B model_card
mkdir -p data/opus-flores && mv opus-flores.zip data

cd data
unzip  opus-flores.zip
mv *-en opus-flores
cd ../

pip3 install -r requirements.txt
pip3 install accelerate==0.29.1 datasets==2.18.0 trl==0.8.1 unbabel-comet tensorboardX tensorboard sacrebleu[ja] sacrebleu[ko]
cd transformers-4.39.2 
pip3 install -e .
cd ..
