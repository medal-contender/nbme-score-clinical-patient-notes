import ast
import random
import torch
import string
import munch
import yaml
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def id_generator(size=12, chars=string.ascii_lowercase + string.digits):
    '''
        학습 버전을 구분짓기 위한 해시를 생성합니다.
    '''
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))


def get_train(csv_path='../input/nbme-score-clinical-patient-notes/train.csv'):
    train = pd.read_csv(csv_path)
    train['annotation'] = train['annotation'].apply(ast.literal_eval)
    train['location'] = train['location'].apply(ast.literal_eval)
    features = pd.read_csv(
        '../input/nbme-score-clinical-patient-notes/features.csv')
    features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
    patient_notes = pd.read_csv(
        '../input/nbme-score-clinical-patient-notes/patient_notes.csv')

    train = train.merge(features, on=['feature_num', 'case_num'], how='left')
    train = train.merge(patient_notes, on=['pn_num', 'case_num'], how='left')

    return train, features, patient_notes

def get_folded_dataframe(merged, n_splits):
    skf = StratifiedKFold(n_splits=n_splits)
    merged["stratify_on"] = merged["case_num"].astype(str) + merged["feature_num"].astype(str)
    merged["fold"] = -1

    for fold, (_, valid_idx) in enumerate(skf.split(merged["id"], y=merged["stratify_on"])):
        merged.loc[valid_idx, "fold"] = fold
    
    return merged

def get_best_model(save_dir):
    model_list = glob(save_dir + '/*.bin')
    best_loss = float("inf")
    best_model = None

    for model in model_list:
        loss = float(model.split('_')[-1][:-4])
        if loss <= best_loss:
            best_loss = loss
            best_model = model

    return best_model

def get_maxlen(features, patient_notes, cfg):
    result = 0
    for text_col in ['feature_text']:
        text_len = []
        tk0 = tqdm(features[text_col].fillna("").values,
                   total=len(features), desc=f'get max length for {text_col}')
        for text in tk0:
            length = len(
                cfg.tokenizer(text, add_special_tokens=False)['input_ids'])
            text_len.append(length)
        result += max(text_len)

    for text_col in ['pn_history']:
        text_len = []
        tk0 = tqdm(patient_notes[text_col].fillna("").values,
                   total=len(patient_notes), desc=f'get max length for {text_col}')
        for text in tk0:
            length = len(
                cfg.tokenizer(text, add_special_tokens=False)['input_ids'])
            text_len.append(length)
        result += max(text_len)

    # cls & sep & sep
    return result + 3

class ConfigManager(object):
    def __init__(self, args):

        self.config_file = args.config_file
        self.cfg = self.load_yaml(args.config_file)
        self.cfg = munch.munchify(self.cfg)
        self.cfg.config_file = args.config_file
        if args.train:
            self.cfg.training_keyword = args.training_keyword

    def load_yaml(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.full_load(f)

        return data

