import itertools
import time
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from medal_contender.dataset import TrainDataset, get_location_predictions, calculate_char_CV
from medal_contender.model import NBMEModel

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_fn(dataframe, CFG, fold, save_dir):
    
    # get dataloader
    train = dataframe[dataframe["fold"] != fold].reset_index(drop=True)
    valid = dataframe[dataframe["fold"] == fold].reset_index(drop=True)
    train_ds = TrainDataset(CFG, train)
    valid_ds = TrainDataset(CFG, valid)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=CFG.train_param.batch_size, 
                                           pin_memory=True, shuffle=True, drop_last=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=CFG.train_param.batch_size * 2, 
                                           pin_memory=True, shuffle=False, drop_last=False)
    
    # get model
    model = NBMEModel(CFG, config_path=None, pretrained=True).to(CFG.model_param.device)
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(CFG.train_param.lr),
            weight_decay=float(CFG.train_param.weight_decay)
        )
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    
    history = {"train": [], "valid": []}
    best_loss = np.inf
    
    torch.cuda.empty_cache()

    # training
    for epoch in range(CFG.train_param.epochs):
        model.train()
        train_loss = AverageMeter()
        pbar = tqdm(train_dl)
        for i, batch in enumerate(pbar):
            input_ids = batch[0].to(CFG.model_param.device)
            attention_mask = batch[1].to(CFG.model_param.device)
            labels = batch[2].to(CFG.model_param.device)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss = torch.masked_select(loss, labels > -1).mean()
            loss /= CFG.train_param.gradient_accumulation_steps
            loss.backward()
            if (i+1) % CFG.train_param.gradient_accumulation_steps == 0: 
                optimizer.step()
                optimizer.zero_grad()
            train_loss.update(val=loss.item(), n=len(input_ids))
            pbar.set_postfix(Loss=train_loss.avg)
        print(f"EPOCH: {epoch} train loss: {train_loss.avg}")
        history["train"].append(train_loss.avg)
        
        # evaluation
        model.eval()
        valid_loss = AverageMeter()
        pbar = tqdm(valid_dl)
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                input_ids = batch[0].to(CFG.model_param.device)
                attention_mask = batch[1].to(CFG.model_param.device)
                labels = batch[2].to(CFG.model_param.device)
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                loss = torch.masked_select(loss, labels > -1).mean()
                valid_loss.update(val=loss.item(), n=len(input_ids))
                pbar.set_postfix(Loss=valid_loss.avg)
        print(f"EPOCH: {epoch} valid loss: {valid_loss.avg}")
        history["valid"].append(valid_loss.avg)

        # save model
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), f"{save_dir}/nbme_{fold}.pth")
        
    # evaluation summary
    model.load_state_dict(torch.load( f"{save_dir}/nbme_{fold}.pth", map_location = CFG.model_param.device))
    model.eval()
    preds = []
    offsets = []
    seq_ids = []
    lbls = []
    with torch.no_grad():
        for batch in tqdm(valid_dl):
            input_ids = batch[0].to(CFG.model_param.device)
            attention_mask = batch[1].to(CFG.model_param.device)
            labels = batch[2].to(CFG.model_param.device)
            offset_mapping = batch[3]
            sequence_ids = batch[4]
            logits = model(input_ids, attention_mask)
            preds.append(logits.cpu().numpy())
            offsets.append(offset_mapping.numpy())
            seq_ids.append(sequence_ids.numpy())
            lbls.append(labels.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    offsets = np.concatenate(offsets, axis=0)
    seq_ids = np.concatenate(seq_ids, axis=0)
    lbls = np.concatenate(lbls, axis=0)
    location_preds = get_location_predictions(preds, offsets, seq_ids, test=False)
    score = calculate_char_CV(location_preds, offsets, seq_ids, lbls)
    print(f"Fold: {fold} CV score: {score}")
