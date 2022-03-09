import os
import gc
import warnings
import wandb
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AdamW
from collections import defaultdict
from colorama import Fore, Style

from medal_contender.utils import id_generator, set_seed, get_train, ConfigManager, get_maxlen, get_folded_dataframe
from medal_contender.preprocessing import preprocessing_incorrect
from medal_contender.configs import BERT_MODEL_LIST
from medal_contender.model import NBMEModel, fetch_scheduler
from medal_contender.train import train_fn, valid_fn, get_location_predictions, calculate_char_CV
from medal_contender.dataset import prepare_loaders_qa_task


red_font = Fore.RED
blue_font = Fore.BLUE
yellow_font = Fore.YELLOW
reset_all = Style.RESET_ALL

# 경고 억제
warnings.filterwarnings("ignore")

# CUDA가 구체적인 에러를 보고하도록 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def run_training(
    model,
    optimizer,
    scheduler,
    fold,
    save_dir,
    train_loader,
    valid_loader,
    run,
    CFG,
):

    # 자동으로 Gradients를 로깅
    wandb.watch(model, log_freq=100)

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_score = 0.
    history = defaultdict(list)
    best_file = None
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(CFG.train_param.epochs):
        gc.collect()
        # train
        avg_loss = train_fn(
            CFG, fold, train_loader, model, criterion,
            optimizer, epoch, scheduler, CFG.model_param.device
        )

        # eval
        avg_val_loss, preds, offsets, seq_ids, lbls  = valid_fn(
            CFG, valid_loader, model, criterion, epoch, scheduler, CFG.model_param.device)

        # scoring
        location_preds = get_location_predictions(preds, offsets, seq_ids, test=False)

        score = calculate_char_CV(location_preds, offsets, seq_ids, lbls)
        print(f"Validation Score: {score}")

        history['Train Loss'].append(avg_loss)
        history['Valid Loss'].append(avg_val_loss)

        # Loss 로깅
        wandb.log({
            "Train Loss": avg_loss,
            "Valid Loss": avg_val_loss,
            "Valid Score": score,
            "Valid Pred": preds
        })

        # 베스트 모델 저장
        if score > best_score:
            print(
                f"{blue_font}Find Best Score ({best_score} ---> {score})")
            best_score = score
            # 이전 베스트 모델 삭제
            if best_file is None:
                best_file = f'{save_dir}/[{cfg.training_keyword.upper()}]_SCHEDULER_{cfg.model_param.scheduler}_FOLD_{fold}_EPOCH_{epoch}_LOSS_{best_score:.4f}.pth'
            else:
                os.remove(best_file)
                best_file = f'{save_dir}/[{cfg.training_keyword.upper()}]_SCHEDULER_{cfg.model_param.scheduler}_FOLD_{fold}_EPOCH_{epoch}_LOSS_{best_score:.4f}.pth'

            run.summary["Best Loss"] = best_score
            PATH = f"{save_dir}/[{cfg.training_keyword.upper()}]_SCHEDULER_{cfg.model_param.scheduler}_FOLD_{fold}_EPOCH_{epoch}_LOSS_{best_score:.4f}.pth"
            # 모델 저장
            torch.save(model.state_dict(), PATH)
            print(f"{red_font} Best Score {best_score} Model Saved{reset_all}")

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_score))

    return history


def main(CFG):
    print(BERT_MODEL_LIST[cfg.model_param.model_name])
    CFG.tokenizer = AutoTokenizer.from_pretrained(
        f"../models/{BERT_MODEL_LIST[cfg.model_param.model_name]}"
    )
    CFG.group = f'{CFG.program_param.project_name}.{CFG.model_param.model_name}.{CFG.training_keyword}'

    wandb.login(key=CFG.program_param.wandb_key)

    set_seed(CFG.program_param.seed)

    HASH_NAME = id_generator(size=12)

    # 모델 저장 경로
    root_save_dir = '../checkpoint'
    save_dir = os.path.join(root_save_dir, CFG.model_param.model_name)
    os.makedirs(save_dir, exist_ok=True)

    # 데이터프레임
    train_df, features, patient_notes = get_train()

    # 데이터 전처리
    train_df = preprocessing_incorrect(train_df)
    
    # K Fold
    train_df = get_folded_dataframe(
        train_df,
        CFG.train_param.n_fold,
        CFG.train_param.kfold_type,
    )

    # Debug
    if CFG.train_param.debug:
        train_df = train_df.sample(n=500).reset_index(drop=True)
        CFG.max_len = 466
        CFG.train_param.epochs = 1
    else:
        CFG.max_len = get_maxlen(features, patient_notes, CFG)
    
    
    # 학습 진행
    for fold in range(0, CFG.train_param.n_fold):

        print(f"{yellow_font}====== Fold: {fold} ======{reset_all}")

        run = wandb.init(
            project=CFG.program_param.project_name,
            config=CFG,
            job_type='Train',
            group=CFG.group,
            tags=[
                CFG.model_param.model_name, HASH_NAME, CFG.train_param.loss
            ],
            name=f'{HASH_NAME}-fold-{fold}',
            anonymous='must'
        )

        # get dataloader
        train_loader, valid_loader, train_fold_len = prepare_loaders_qa_task(train_df, CFG, fold)
        
        # get model
        model = NBMEModel(CFG, config_path=None, pretrained=True).to(CFG.model_param.device)
        optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(CFG.train_param.lr),
                weight_decay=float(CFG.train_param.weight_decay)
            )

        num_train_steps = int(
            train_fold_len / CFG.train_param.batch_size * CFG.train_param.epochs)
        scheduler = fetch_scheduler(optimizer, CFG, num_train_steps=num_train_steps)

        history = run_training(
            model,
            optimizer,
            scheduler,
            fold,
            save_dir,
            train_loader,
            valid_loader,
            run,
            CFG
        )

        run.finish()

        if fold == CFG.train_param.n_fold-1:
            config_path = f"{save_dir}/[{CFG.training_keyword.upper()}]_SCHEDULER_{CFG.model_param.scheduler}_config.pth"
            torch.save(model.config, config_path)

        del model, history, train_loader, valid_loader
        _ = gc.collect()
        print()


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
