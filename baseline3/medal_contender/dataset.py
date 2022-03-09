from ast import literal_eval
from dataclasses import dataclass
from functools import partial

import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import DataCollatorForTokenClassification
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def loc_list_to_ints(loc_list):
    to_return = []

    for loc_str in loc_list:
        loc_strs = loc_str.split(";")

        for loc in loc_strs:
            start, end = loc.split()
            to_return.append((int(start), int(end)))

    return to_return


def process_feature_text(text):
    return text.replace("-", " ")


def tokenize_and_add_labels(example, tokenizer, CFG):

    tokenized_inputs = tokenizer(
        example["feature_text"],
        example["text"],
        truncation="only_second",
        max_length=CFG.data_param.max_seq_length,
        padding=True,
        return_offsets_mapping=True
    )

    # labels should be float
    labels = [0.0]*len(tokenized_inputs["input_ids"])
    tokenized_inputs["locations"] = loc_list_to_ints(example["loc_list"])
    tokenized_inputs["sequence_ids"] = tokenized_inputs.sequence_ids()

    for idx, (seq_id, offsets) in enumerate(zip(tokenized_inputs["sequence_ids"], tokenized_inputs["offset_mapping"])):
        if seq_id is None or seq_id == 0:
            # don't calculate loss on question part or special tokens
            labels[idx] = -100.0
            continue

        exit = False
        token_start, token_end = offsets
        for feature_start, feature_end in tokenized_inputs["locations"]:
            if exit:
                break
            if token_start <= feature_start < token_end or token_start < feature_end <= token_end or feature_start <= token_start < feature_end:
                labels[idx] = 1.0  # labels should be float
                exit = True

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def get_train(csv_path='../input/nbme-score-clinical-patient-notes/train.csv'):
    train = pd.read_csv(csv_path)
    train['anno_list'] = train['annotation'].apply(literal_eval)
    train['loc_list'] = train['location'].apply(literal_eval)
    features = pd.read_csv(
        '../input/nbme-score-clinical-patient-notes/features.csv')
    features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
    patient_notes = pd.read_csv(
        '../input/nbme-score-clinical-patient-notes/patient_notes.csv')

    return train, features, patient_notes


def preprocess_train(CFG):
    train_df, feats_df, notes_df = get_train()
    # # Stratified KFold
    #
    # Without any leaks this time ;)
    # Thanks @theoviel https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/305599#1678215

    skf = StratifiedKFold(
        n_splits=CFG.data_param.k_folds, random_state=CFG.train_param.seed,
        shuffle=True
    )

    notes_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(notes_df, y=notes_df["case_num"])):
        notes_df.loc[val_idx, "fold"] = fold

    merged = train_df.merge(notes_df, how="left")
    merged = merged.merge(feats_df, how="left")

    # ## Correcting some annotations
    #
    # Huge shoutout to @yasufuminakama for this work: https://www.kaggle.com/yasufuminakama/nbme-deberta-base-baseline-train?scriptVersionId=87264998&cellId=17

    # incorrect annotations
    merged.loc[338, "anno_list"] = '["father heart attack"]'
    merged.loc[338, "loc_list"] = '["764 783"]'

    merged.loc[621, "anno_list"] = '["for the last 2-3 months", "over the last 2 months"]'
    merged.loc[621, "loc_list"] = '["77 100", "398 420"]'

    merged.loc[655, "anno_list"] = '["no heat intolerance", "no cold intolerance"]'
    merged.loc[655, "loc_list"] = '["285 292;301 312", "285 287;296 312"]'

    merged.loc[1262, "anno_list"] = '["mother thyroid problem"]'
    merged.loc[1262, "loc_list"] = '["551 557;565 580"]'

    merged.loc[1265, "anno_list"] = '[\'felt like he was going to "pass out"\']'
    merged.loc[1265, "loc_list"] = '["131 135;181 212"]'

    merged.loc[1396, "anno_list"] = '["stool , with no blood"]'
    merged.loc[1396, "loc_list"] = '["259 280"]'

    merged.loc[1591, "anno_list"] = '["diarrhoe non blooody"]'
    merged.loc[1591, "loc_list"] = '["176 184;201 212"]'

    merged.loc[1615, "anno_list"] = '["diarrhea for last 2-3 days"]'
    merged.loc[1615, "loc_list"] = '["249 257;271 288"]'

    merged.loc[1664, "anno_list"] = '["no vaginal discharge"]'
    merged.loc[1664, "loc_list"] = '["822 824;907 924"]'

    merged.loc[1714, "anno_list"] = '["started about 8-10 hours ago"]'
    merged.loc[1714, "loc_list"] = '["101 129"]'

    merged.loc[1929, "anno_list"] = '["no blood in the stool"]'
    merged.loc[1929, "loc_list"] = '["531 539;549 561"]'

    merged.loc[2134, "anno_list"] = '["last sexually active 9 months ago"]'
    merged.loc[2134, "loc_list"] = '["540 560;581 593"]'

    merged.loc[2191, "anno_list"] = '["right lower quadrant pain"]'
    merged.loc[2191, "loc_list"] = '["32 57"]'

    merged.loc[2553, "anno_list"] = '["diarrhoea no blood"]'
    merged.loc[2553, "loc_list"] = '["308 317;376 384"]'

    merged.loc[3124, "anno_list"] = '["sweating"]'
    merged.loc[3124, "loc_list"] = '["549 557"]'

    merged.loc[3858, "anno_list"] = '["previously as regular", "previously eveyr 28-29 days", "previously lasting 5 days", "previously regular flow"]'
    merged.loc[3858, "loc_list"] = '["102 123", "102 112;125 141", "102 112;143 157", "102 112;159 171"]'

    merged.loc[4373, "anno_list"] = '["for 2 months"]'
    merged.loc[4373, "loc_list"] = '["33 45"]'

    merged.loc[4763, "anno_list"] = '["35 year old"]'
    merged.loc[4763, "loc_list"] = '["5 16"]'

    merged.loc[4782, "anno_list"] = '["darker brown stools"]'
    merged.loc[4782, "loc_list"] = '["175 194"]'

    merged.loc[4908, "anno_list"] = '["uncle with peptic ulcer"]'
    merged.loc[4908, "loc_list"] = '["700 723"]'

    merged.loc[6016, "anno_list"] = '["difficulty falling asleep"]'
    merged.loc[6016, "loc_list"] = '["225 250"]'

    merged.loc[6192, "anno_list"] = '["helps to take care of aging mother and in-laws"]'
    merged.loc[6192, "loc_list"] = '["197 218;236 260"]'

    merged.loc[6380, "anno_list"] = '["No hair changes", "No skin changes", "No GI changes", "No palpitations", "No excessive sweating"]'
    merged.loc[6380, "loc_list"] = '["480 482;507 519", "480 482;499 503;512 519", "480 482;521 531", "480 482;533 545", "480 482;564 582"]'

    merged.loc[6562, "anno_list"] = '["stressed due to taking care of her mother", "stressed due to taking care of husbands parents"]'
    merged.loc[6562, "loc_list"] = '["290 320;327 337", "290 320;342 358"]'

    merged.loc[6862, "anno_list"] = '["stressor taking care of many sick family members"]'
    merged.loc[6862, "loc_list"] = '["288 296;324 363"]'

    merged.loc[7022, "anno_list"] = '["heart started racing and felt numbness for the 1st time in her finger tips"]'
    merged.loc[7022, "loc_list"] = '["108 182"]'

    merged.loc[7422, "anno_list"] = '["first started 5 yrs"]'
    merged.loc[7422, "loc_list"] = '["102 121"]'

    merged.loc[8876, "anno_list"] = '["No shortness of breath"]'
    merged.loc[8876, "loc_list"] = '["481 483;533 552"]'

    merged.loc[9027, "anno_list"] = '["recent URI", "nasal stuffines, rhinorrhea, for 3-4 days"]'
    merged.loc[9027, "loc_list"] = '["92 102", "123 164"]'

    merged.loc[9938, "anno_list"] = '["irregularity with her cycles", "heavier bleeding", "changes her pad every couple hours"]'
    merged.loc[9938, "loc_list"] = '["89 117", "122 138", "368 402"]'

    merged.loc[9973, "anno_list"] = '["gaining 10-15 lbs"]'
    merged.loc[9973, "loc_list"] = '["344 361"]'

    merged.loc[10513, "anno_list"] = '["weight gain", "gain of 10-16lbs"]'
    merged.loc[10513, "loc_list"] = '["600 611", "607 623"]'

    merged.loc[11551, "anno_list"] = '["seeing her son knows are not real"]'
    merged.loc[11551, "loc_list"] = '["386 400;443 461"]'

    merged.loc[11677, "anno_list"] = '["saw him once in the kitchen after he died"]'
    merged.loc[11677, "loc_list"] = '["160 201"]'

    merged.loc[12124, "anno_list"] = '["tried Ambien but it didnt work"]'
    merged.loc[12124, "loc_list"] = '["325 337;349 366"]'

    merged.loc[12279, "anno_list"] = '["heard what she described as a party later than evening these things did not actually happen"]'
    merged.loc[12279, "loc_list"] = '["405 459;488 524"]'

    merged.loc[12289, "anno_list"] = '["experienced seeing her son at the kitchen table these things did not actually happen"]'
    merged.loc[12289, "loc_list"] = '["353 400;488 524"]'

    merged.loc[13238, "anno_list"] = '["SCRACHY THROAT", "RUNNY NOSE"]'
    merged.loc[13238, "loc_list"] = '["293 307", "321 331"]'

    merged.loc[13297, "anno_list"] = '["without improvement when taking tylenol", "without improvement when taking ibuprofen"]'
    merged.loc[13297, "loc_list"] = '["182 221", "182 213;225 234"]'

    merged.loc[13299, "anno_list"] = '["yesterday", "yesterday"]'
    merged.loc[13299, "loc_list"] = '["79 88", "409 418"]'

    merged.loc[13845, "anno_list"] = '["headache global", "headache throughout her head"]'
    merged.loc[13845, "loc_list"] = '["86 94;230 236", "86 94;237 256"]'

    merged.loc[14083, "anno_list"] = '["headache generalized in her head"]'
    merged.loc[14083, "loc_list"] = '["56 64;156 179"]'

    merged["anno_list"] = [literal_eval(x) if isinstance(
        x, str) else x for x in merged["anno_list"]]
    merged["loc_list"] = [literal_eval(x) if isinstance(
        x, str) else x for x in merged["loc_list"]]

    merged = merged[merged["anno_list"].map(
        len) != 0].copy().reset_index(drop=True)

    merged["feature_text"] = [
        process_feature_text(x) for x in merged["feature_text"]]

    return merged[["id", "case_num", "pn_num", "feature_num", "loc_list",
                   "pn_history", "feature_text", "fold"]]


def get_tokenized_dataset(tokenizer, CFG):
    merged = preprocess_train(CFG)
    
    # Debug
    if CFG.train_param.debug:
        merged = merged.sample(n=500).reset_index(drop=True)
        CFG.train_param.num_train_epochs = 1

    dataset = Dataset.from_pandas(merged)
    dataset = dataset.rename_column("pn_history", "text")
    tokenized_dataset = dataset.map(
        partial(tokenize_and_add_labels, tokenizer=tokenizer, CFG=CFG),
        desc="Tokenizing and adding labels"
    )
    return tokenized_dataset


@dataclass
class CustomDataCollator(DataCollatorForTokenClassification):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Have to modify to make label tensors float and not int.
    """

    tokenizer = PreTrainedTokenizerBase
    padding = True
    max_length = None
    pad_to_multiple_of = None
    label_pad_token_id = -100
    return_tensors = "pt"

    def torch_call(self, features):
        batch = super().torch_call(features)
        label_name = "label" if "label" in features[0].keys() else "labels"

        batch[label_name] = torch.tensor(
            batch[label_name], dtype=torch.float32)

        return batch
