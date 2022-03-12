import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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


def prepare_loaders_qa_task(dataframe, CFG, fold):
    train = dataframe[dataframe["fold"] != fold].reset_index(drop=True)
    valid = dataframe[dataframe["fold"] == fold].reset_index(drop=True)
    train_fold_len = len(train)
    
    train_ds = TrainDataset(CFG, train)
    valid_ds = TrainDataset(CFG, valid)
    train_loader = DataLoader(train_ds, batch_size=CFG.train_param.batch_size, 
                                        pin_memory=True, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=CFG.train_param.batch_size * 2, 
                                        pin_memory=True, shuffle=False, drop_last=False)

    return train_loader, valid_loader, train_fold_len