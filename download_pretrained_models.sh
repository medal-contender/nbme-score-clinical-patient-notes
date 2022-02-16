# 디렉토리 생성
mkdir models
cd ./models

# 모델 레포지토리 클론
apt-get install git-lfs
# HateBERT
git clone https://huggingface.co/GroNLP/hateBERT
cd ./hateBERT
GIT_LFS_SKIP_SMUDGE=1
cd ..
# RoBERTa-Base
git clone https://huggingface.co/roberta-base
cd ./roberta-base
GIT_LFS_SKIP_SMUDGE=1
cd ..
# RoBERTa-Large
git clone https://huggingface.co/roberta-large
cd ./roberta-large
GIT_LFS_SKIP_SMUDGE=1
cd ..
# DistilBERT
git clone https://huggingface.co/distilbert-base-cased
cd ./distilbert-base-cased
GIT_LFS_SKIP_SMUDGE=1
cd ..
# Electra
git clone https://huggingface.co/google/electra-base-discriminator
cd ./electra-base-discriminator
GIT_LFS_SKIP_SMUDGE=1
cd ..
# LUKE
git clone https://huggingface.co/studio-ousia/luke-base
cd ./luke-base
GIT_LFS_SKIP_SMUDGE=1
cd ..
# DeBERTa
git clone https://huggingface.co/microsoft/deberta-v3-base
cd ./deberta-v3-base
GIT_LFS_SKIP_SMUDGE=1
cd ..
# Bigbird-RoBERTa
git clone https://huggingface.co/google/bigbird-roberta-base
cd ./bigbird-roberta-base
GIT_LFS_SKIP_SMUDGE=1
cd ..
# T5
git clone https://huggingface.co/t5-base
cd ./t5-base
GIT_LFS_SKIP_SMUDGE=1
cd ..
# BERT
git clone https://huggingface.co/bert-base-cased
cd ./bert-base-cased
GIT_LFS_SKIP_SMUDGE=1
cd ..
# ToxicBERT
git clone https://huggingface.co/unitary/toxic-bert
cd ./toxic-bert
GIT_LFS_SKIP_SMUDGE=1
cd ..
# Toxic RoBERTa
git clone https://huggingface.co/unitary/unbiased-toxic-roberta
cd ./unbiased-toxic-roberta
GIT_LFS_SKIP_SMUDGE=1
cd ..
# Funnel
git clone https://huggingface.co/funnel-transformer/intermediate
cd ./intermediate
GIT_LFS_SKIP_SMUDGE=1
cd ..
# Muppet RoBERTa
git clone https://huggingface.co/facebook/muppet-roberta-base
cd ./muppet-roberta-base
GIT_LFS_SKIP_SMUDGE=1
cd ..
# MPNet
git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2
cd ./all-mpnet-base-v2
GIT_LFS_SKIP_SMUDGE=1
cd ..
# DistilBART
git clone https://huggingface.co/sshleifer/distilbart-cnn-6-6
mv ./distilbart-cnn-6-6 ./distilbart-cnn-6-6-large
cd ./distilbart-cnn-6-6-large
GIT_LFS_SKIP_SMUDGE=1
cd ..