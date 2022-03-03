import ast
import numpy as np
from itertools import chain
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loc_list_to_ints(loc_list):
    to_return = []
    for loc_str in loc_list:
        loc_strs = loc_str.split(";")
        for loc in loc_strs:
            start, end = loc.split()
            to_return.append((int(start), int(end)))
    return to_return

def tokenize_and_add_labels(CFG, feature_text, pn_history, location):
    tokenized_inputs = CFG.tokenizer(
        feature_text,      # question
        pn_history,        # content
        truncation="only_second",
        max_length=CFG.max_len,
        padding="max_length",
        return_offsets_mapping=True
    )
    labels = [0.0] * len(tokenized_inputs["input_ids"])
    tokenized_inputs["location_int"] = loc_list_to_ints(location)
    tokenized_inputs["sequence_ids"] = tokenized_inputs.sequence_ids()
    
    for idx, (seq_id, offsets) in enumerate(zip(tokenized_inputs["sequence_ids"], tokenized_inputs["offset_mapping"])):
        if seq_id is None or seq_id == 0:     # seq_id == None: special tokens | seq_id == 0: question
            labels[idx] = -100
            continue
        exit = False
        token_start, token_end = offsets
        for feature_start, feature_end in tokenized_inputs["location_int"]:
            if exit:
                break
            if token_start >= feature_start and token_end <= feature_end:
                labels[idx] = 1.0
                exit = True
    tokenized_inputs["labels"] = labels
    
    return tokenized_inputs

class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values
        self.locations = df['location'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        tokenized = tokenize_and_add_labels(self.cfg,
                             self.feature_texts[item],
                             self.pn_historys[item],
                             self.locations[item],
                             )
        input_ids = np.array(tokenized["input_ids"])
        attention_mask = np.array(tokenized["attention_mask"])
        labels = np.array(tokenized["labels"])
        offset_mapping = np.array(tokenized["offset_mapping"])
        sequence_ids = np.array(tokenized["sequence_ids"]).astype("float16")

        return input_ids, attention_mask, labels, offset_mapping, sequence_ids

def get_location_predictions(preds, offset_mapping, sequence_ids, test = False):
    all_predictions = []
    for pred, offsets, seq_ids in zip(preds, offset_mapping, sequence_ids):
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
                if test:
                    current_preds.append(f"{start_idx} {end_idx}")
                else:
                    current_preds.append((start_idx, end_idx))
                start_idx = None
        if test:
            all_predictions.append("; ".join(current_preds))
        else:
            all_predictions.append(current_preds)
    return all_predictions


def calculate_char_CV(predictions, offset_mapping, sequence_ids, labels):
    all_labels = []
    all_preds = []
    for preds, offsets, seq_ids, labels in zip(predictions, offset_mapping, sequence_ids, labels):
        num_chars = max(list(chain(*offsets)))
        char_labels = np.zeros((num_chars))
        for o, s_id, label in zip(offsets, seq_ids, labels):
            if s_id is None or s_id == 0:
                continue
            if int(label) == 1:
                char_labels[o[0]:o[1]] = 1
        char_preds = np.zeros((num_chars))
        for start_idx, end_idx in preds:
            char_preds[start_idx:end_idx] = 1
        all_labels.extend(char_labels)
        all_preds.extend(char_preds)
    results = precision_recall_fscore_support(all_labels, all_preds, average = "binary")
    return {
        "precision": results[0],
        "recall": results[1],
        "f1": results[2]
    }

def compute_metrics(p):
    predictions, y_true = p
    y_true = y_true.astype(int)
    y_pred = [
        [int(p > 0.5) for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, y_true)
    ]
    y_true = [
        [l for l in label if l != -100] for label in y_true
    ]
    results = precision_recall_fscore_support(list(chain(*y_true)), list(chain(*y_pred)), average = "binary")
    return {
        "token_precision": results[0],
        "token_recall": results[1],
        "token_f1": results[2]
    }
