import os
import gc
import warnings
import wandb
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AdamW
from collections import defaultdict
from medal_contender.utils import (
    id_generator, set_seed, get_train, ConfigManager, get_maxlen, get_folded_dataframe
)
from medal_contender.preprocessing import preprocessing_incorrect
from medal_contender.configs import BERT_MODEL_LIST
from medal_contender.model import NBMEModel
from medal_contender.train import train_fn
from colorama import Fore, Style

red_font = Fore.RED
blue_font = Fore.BLUE
yellow_font = Fore.YELLOW
reset_all = Style.RESET_ALL

# 경고 억제
warnings.filterwarnings("ignore")

# CUDA가 구체적인 에러를 보고하도록 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def run_training(
    dataframe,
    CFG,
):

    for fold in range(CFG.train_param.n_fold):
        train_fn(dataframe, CFG, fold)

    return

def main(CFG):
    CFG.tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_LIST[CFG.model_param.model_name]
    )
    CFG.group = f'{CFG.program_param.project_name}.{CFG.model_param.model_name}.{CFG.training_keyword}'

    wandb.login(key=CFG.program_param.wandb_key)

    set_seed(CFG.program_param.seed)

    HASH_NAME = id_generator(size=12)

    # 모델 저장 경로
    root_save_dir = '../checkpoint'
    save_dir = os.path.join(root_save_dir, CFG.model_param.model_name)
    os.makedirs(save_dir, exist_ok=True)

    # 데이터 경로
    # root_data_dir = '../input/'+cfg.data_param.dir_name.replace("_","-")
    # train_csv = os.path.join(root_data_dir,cfg.data_param.train_file_name)

    # 데이터프레임
    train_df, features, patient_notes = get_train()

    # 데이터 전처리
    train_df = preprocessing_incorrect(train_df)
    
    # K Fold
    train_df = get_folded_dataframe(
        train_df,
        CFG.train_param.n_fold,
    )

    CFG.max_len = get_maxlen(features, patient_notes, CFG)
    # train_df.to_csv('../input/train_df.csv')
    
    run_training(
        train_df,
        CFG,
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Type Name Of Config File To Use."
    )
    parser.add_argument(
        "--train",
        action='store_true',
        help="Toggle On If Model Is On Training."
    )
    parser.add_argument(
        "--training-keyword",
        type=str,
        default='hyperparameter_tuning',
        help="Type Keyword Of This Training."
    )

    args = parser.parse_args()

    cfg = ConfigManager(args).cfg
    main(cfg)
