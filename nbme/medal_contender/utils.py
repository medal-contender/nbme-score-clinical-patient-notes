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
from sklearn.model_selection import GroupKFold

# For Group K-Fold Strategy


class UnionFind():
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.parents[x] > self.parents[y]:
            x, y = y, x
        self.parents[x] += self.parents[y]
        self.parents[y] = x


def micro_f1(preds, truths):
    """
    Micro f1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds)


def spans_to_binary(spans, length=None):
    """
    Converts spans to a binary array indicating whether each character is in the span.

    Args:
        spans (list of lists of two ints): Spans.

    Returns:
        np array [length]: Binarized spans.
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1
    return binary


def span_micro_f1(preds, truths):
    """
    Micro f1 on spans.

    Args:
        preds (list of lists of two ints): Prediction spans.
        truths (list of lists of two ints): Ground truth spans.

    Returns:
        float: f1 score.
    """
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(np.max(pred) if len(pred) else 0,
                     np.max(truth) if len(truth) else 0)
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))
    return micro_f1(bin_preds, bin_truths)


def create_labels_for_scoring(df):
    # example: ['0 1', '3 4'] -> ['0 1; 3 4']
    df['location_for_create_labels'] = [ast.literal_eval(f'[]')] * len(df)
    for i in range(len(df)):
        lst = df.loc[i, 'location']
        if lst:
            new_lst = ';'.join(lst)
            df.loc[i, 'location_for_create_labels'] = ast.literal_eval(
                f'[["{new_lst}"]]')
    # create labels
    truths = []
    for location_list in df['location_for_create_labels'].values:
        truth = []
        if len(location_list) > 0:
            location = location_list[0]
            for loc in [s.split() for s in location.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                truth.append([start, end])
        truths.append(truth)
    return truths


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


def get_folded_dataframe(df, n_splits):
    fold = GroupKFold(n_splits=n_splits)
    groups = df['pn_num'].values
    for num, (train_index, val_index) in enumerate(fold.split(df, df['location'], groups)):
        df.loc[val_index, 'fold'] = int(num)
    df['fold'] = df['fold'].astype(int)
    # print(df.groupby('fold').size())
    return df


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


def get_score(y_true, y_pred):
    score = span_micro_f1(y_true, y_pred)
    return score

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
