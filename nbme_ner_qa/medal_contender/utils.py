import ast
import os
import random
import string
from glob import glob
from itertools import chain

import munch
import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm


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


def get_folded_dataframe(df, n_splits, kfold_type):
    if kfold_type == 'skf':
        skf = StratifiedKFold(n_splits=n_splits)
        df["stratify_on"] = df["case_num"].astype(str) + df["feature_num"].astype(str)
        df["fold"] = -1
        for fold, (_, valid_idx) in enumerate(skf.split(df["id"], y=df["stratify_on"])):
            df.loc[valid_idx, "fold"] = fold
        return df

    elif kfold_type == 'group':
        fold = GroupKFold(n_splits=n_splits)
        groups = df['pn_num'].values
        for num, (train_index, val_index) in enumerate(fold.split(df, df['location'], groups)):
            df.loc[val_index, 'fold'] = int(num)
        df['fold'] = df['fold'].astype(int)
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


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def compute_metrics(eval_prediction):
    """
    This only gets the scores at the token level.
    The actual leaderboard score is based at the character level.
    The CV score at the character level is handled in the evaluate
    function of the trainer.
    """
    predictions, y_true = eval_prediction
    predictions = sigmoid(predictions)
    y_true = y_true.astype(int)

    y_pred = [
        [int(p > 0.5) for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, y_true)
    ]

    # Remove ignored index (special tokens)
    y_true = [
        [l for l in label if l != -100]
        for label in y_true
    ]

    results = precision_recall_fscore_support(
        list(chain(*y_true)), list(chain(*y_pred)), average="binary")
    return {
        "token_precision": results[0],
        "token_recall": results[1],
        "token_f1": results[2]
    }


def get_location_predictions(dataset, preds):
    """
    It's easier to run CV if we don't convert predictions into
    the format expected at test time.
    """
    all_predictions = []
    for pred, offsets, seq_ids in zip(preds, dataset["offset_mapping"], dataset["sequence_ids"]):
        pred = sigmoid(pred)
        start_idx = None
        current_preds = []
        for p, o, s_id in zip(pred, offsets, seq_ids):
            if s_id is None or s_id == 0:
                continue

            if p > 0.5:
                if start_idx is None:
                    start_idx = o[0]
                end_idx = o[1]
            elif start_idx is not None:
                current_preds.append((start_idx, end_idx))
                start_idx = None

        if start_idx is not None:
            current_preds.append((start_idx, end_idx))

        all_predictions.append(current_preds)

    return all_predictions


def calculate_char_CV(dataset, predictions):
    """
    Some tokenizers include the leading space as the start of the
    offset_mapping, so there is code to ignore that space.
    """
    all_labels = []
    all_preds = []
    for preds, offsets, seq_ids, labels, text in zip(
        predictions,
        dataset["offset_mapping"],
        dataset["sequence_ids"],
        dataset["labels"],
        dataset["text"]
    ):

        num_chars = max(list(chain(*offsets)))
        char_labels = np.zeros((num_chars))

        for o, s_id, label in zip(offsets, seq_ids, labels):
            if s_id is None or s_id == 0:  # ignore question part of input
                continue
            if int(label) == 1:

                char_labels[o[0]:o[1]] = 1
                if text[o[0]].isspace() and o[0] > 0 and char_labels[o[0]-1] != 1:
                    char_labels[o[0]] = 0

        char_preds = np.zeros((num_chars))

        for start_idx, end_idx in preds:
            char_preds[start_idx:end_idx] = 1
            if text[start_idx].isspace():
                char_preds[start_idx] = 0

        all_labels.extend(char_labels)
        all_preds.extend(char_preds)

    results = precision_recall_fscore_support(
        all_labels, all_preds, average="binary")
    return {
        "precision": results[0],
        "recall": results[1],
        "f1": results[2]
    }
