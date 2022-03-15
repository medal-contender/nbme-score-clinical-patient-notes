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
    id_generator, set_seed, get_train, get_folded_dataframe, ConfigManager,
    get_score, create_labels_for_scoring, get_maxlen
)
from medal_contender.preprocessing import preprocessing_incorrect
from medal_contender.configs import BERT_MODEL_LIST
from medal_contender.dataset import prepare_loaders
from medal_contender.model import NBMEModel, DeepShareModel, fetch_scheduler, get_optimizer_params
from medal_contender.train import (
    train_fn, valid_fn, get_predictions, get_char_probs, get_results
)
from colorama import Fore, Style

# To make tokenizer in new environment, please operate the code below (3 line)
'''
from medal_contender.tokenizer import load_tokenizer
from medal_contender.configs import MAKE_TOKENIZER
load_tokenizer(MAKE_TOKENIZER)
'''

from transformers.models.deberta_v2 import DebertaV2TokenizerFast

red_font = Fore.RED
blue_font = Fore.BLUE
yellow_font = Fore.YELLOW
reset_all = Style.RESET_ALL

# 경고 억제
warnings.filterwarnings("ignore")

# CUDA가 구체적인 에러를 보고하도록 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Tokenizer parallelism을 사용하도록 환경 설정
os.environ["TOKENIZERS_PARALLELISM"] = "true" 


def run_training(
    model,
    optimizer,
    scheduler,
    fold,
    save_dir,
    train_loader,
    valid_loader,
    valid_texts,
    valid_labels,
    valid_fold_len,
    run,
    CFG,
    val_folds
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
        avg_val_loss, predictions = valid_fn(
            CFG, valid_loader, model, criterion, epoch, scheduler, CFG.model_param.device)

        predictions = predictions.reshape((valid_fold_len, CFG.max_len))

        # scoring
        char_probs = get_char_probs(valid_texts, predictions, CFG.tokenizer)
        results = get_results(char_probs, th=0.5)
        preds = get_predictions(results)
        score = get_score(valid_labels, preds)

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
            torch.save({'model': model.state_dict(),
                        'predictions': predictions}, PATH)
            print(f"{red_font} Best Score {best_score} Model Saved{reset_all}")

        print()

    predictions = torch.load(PATH,
                             map_location=torch.device('cpu'))['predictions']
    val_folds[[i for i in range(CFG.max_len)]] = predictions

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_score))

    return history, val_folds


def get_result(oof_df, CFG):
    labels = create_labels_for_scoring(oof_df)
    predictions = oof_df[[i for i in range(CFG.max_len)]].values
    char_probs = get_char_probs(
        oof_df['pn_history'].values, predictions, CFG.tokenizer)
    results = get_results(char_probs, th=0.5)
    preds = get_predictions(results)
    score = get_score(labels, preds)


def main(CFG):
    root_save_dir = '../checkpoint'
    CFG.tokenizer = DebertaV2TokenizerFast.from_pretrained(BERT_MODEL_LIST[CFG.model_param.model_name])

    #tokenizer 저장
    CFG.tokenizer.save_pretrained(os.path.join(root_save_dir, 'get_token'))

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
    train_df = preprocessing_incorrect(train_df, add_correct=CFG.train_param.add_correct)

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
    oof_df_ = pd.DataFrame()
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

        train_loader, valid_loader, valid_texts, valid_labels, valid_fold_len, train_fold_len, val_folds = \
            prepare_loaders(train_df, CFG, fold)
        if CFG.train_param.model_type == "Attention":
            model = NBMEModel(CFG, config_path=None, pretrained=True)
        elif CFG.train_param.model_type == "DeepShareModel":
            model = DeepShareModel(CFG, config_path=None, pretrained=True)
        model.to(CFG.model_param.device)

        # optimizer_parameters = get_optimizer_params(
        #     model,
        #     encoder_lr=CFG.train_param.encoder_lr,
        #     decoder_lr=CFG.train_param.decoder_lr,
        #     weight_decay=CFG.train_param.weight_decay
        # )

        # optimizer = AdamW(
        #     optimizer_parameters,
        #     lr=float(CFG.train_param.lr),
        #     eps=float(CFG.train_param.eps),
        #     # betas=eval(CFG.train_param.betas)
        # )
        optimizer = AdamW(
            model.parameters(),
            lr=float(cfg.train_param.lr),
            weight_decay=float(cfg.train_param.weight_decay)
        )
        num_train_steps = int(
            train_fold_len / CFG.train_param.batch_size * CFG.train_param.epochs)
        scheduler = fetch_scheduler(optimizer, CFG, num_train_steps=num_train_steps)

        history, oof_df = run_training(
            model,
            optimizer,
            scheduler,
            fold,
            save_dir,
            train_loader,
            valid_loader,
            valid_texts,
            valid_labels,
            valid_fold_len,
            run,
            CFG,
            val_folds,
        )

        run.finish()

        oof_df_ = pd.concat([oof_df, oof_df_])

        if fold == CFG.train_param.n_fold-1:
            config_path = f"{save_dir}/[{CFG.training_keyword.upper()}]_SCHEDULER_{CFG.model_param.scheduler}_config.pth"
            torch.save(model.config, config_path)

        del model, history, train_loader, valid_loader
        _ = gc.collect()
        print()

    # Prediction for oof save
    oof_df_ = oof_df_.reset_index(drop=True)
    oof_df_.to_pickle(
        f'{save_dir}/[{CFG.training_keyword.upper()}]_SCHEDULER_{CFG.model_param.scheduler}_oof_df.pkl')


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
