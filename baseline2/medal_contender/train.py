from itertools import chain
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import precision_recall_fscore_support
from medal_contender.utils import span_micro_f1



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

def train_fn(CFG, fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = GradScaler(enabled=CFG.train_param.apex)
    losses = AverageMeter()
    global_step = 0
    dataset_size = 0
    running_loss = 0.0

    tbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, batch in tbar:
        input_ids = batch[0].to(CFG.model_param.device)
        attention_mask = batch[1].to(CFG.model_param.device)
        labels = batch[2].to(CFG.model_param.device)
        batch_size = labels.size(0)
        with autocast(enabled=CFG.train_param.apex):
            y_preds = model(input_ids, attention_mask)
        loss = criterion(y_preds, labels)
        loss = torch.masked_select(loss, labels > -1).mean()
        if CFG.train_param.gradient_accumulation_steps > 1:
            loss = loss / CFG.train_param.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.train_param.max_grad_norm)
        if (step + 1) % CFG.train_param.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.train_param.batch_scheduler:
                if not CFG.model_param.scheduler == 'rlrp':
                    scheduler.step()
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        tbar.set_postfix(
            Epoch=epoch,
            Train_Loss=epoch_loss,
            Grad_Norm=grad_norm.item(),
            LR=optimizer.param_groups[0]['lr']
        )
    return losses.avg

def valid_fn(CFG, valid_loader, model, criterion, epoch, scheduler, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    offsets = []
    seq_ids = []
    lbls = []
    dataset_size = 0
    running_loss = 0.0
    tbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    
    for step, batch in tbar:
        input_ids = batch[0].to(CFG.model_param.device)
        attention_mask = batch[1].to(CFG.model_param.device)
        labels = batch[2].to(CFG.model_param.device)
        
        offset_mapping = batch[3]
        sequence_ids = batch[4]
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(input_ids, attention_mask)
        loss = criterion(y_preds, labels)
        loss = torch.masked_select(loss, labels > -1).mean()
        if CFG.train_param.gradient_accumulation_steps > 1:
            loss = loss / CFG.train_param.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)

        preds.append(y_preds.cpu().numpy())
        offsets.append(offset_mapping.numpy())
        seq_ids.append(sequence_ids.numpy())
        lbls.append(labels.cpu().numpy())
        

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        tbar.set_postfix(
            Epoch=epoch,
            Valid_Loss=epoch_loss,
        )
        if CFG.model_param.scheduler == 'rlrp':
            scheduler.step(epoch_loss)
        
    preds = np.concatenate(preds, axis=0)
    offsets = np.concatenate(offsets, axis=0)
    seq_ids = np.concatenate(seq_ids, axis=0)
    lbls = np.concatenate(lbls, axis=0)
    '''
    pred shape:  (184, 466)
    offsets shape:  (184, 466, 2)
    seq_ids shape:  (184, 466)
    '''

    return losses.avg, preds, offsets, seq_ids, lbls


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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
    # score = span_micro_f1(all_labels, all_preds)
    score = precision_recall_fscore_support(all_labels, all_preds, average = "binary")
    result = {
        "precision": score[0],
        "recall": score[1],
        "f1": score[2]
    }
    return result['f1'] # f1 score