import argparse
import shutil
import os
import warnings

import torch
from colorama import Fore, Style
from transformers import AutoConfig, AutoTokenizer, TrainingArguments

import wandb
from medal_contender.configs import BERT_MODEL_LIST
from medal_contender.dataset import CustomDataCollator, get_tokenized_dataset
from medal_contender.model import get_model
from medal_contender.train import NBMETrainer
from medal_contender.utils import ConfigManager, compute_metrics, id_generator

red_font = Fore.RED
blue_font = Fore.BLUE
yellow_font = Fore.YELLOW
reset_all = Style.RESET_ALL

# 경고 억제
warnings.filterwarnings("ignore")

# CUDA가 구체적인 에러를 보고하도록 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main(CFG):
    hash_name = id_generator(size=12)
    CFG.group = f'{CFG.program_param.project_name}.{CFG.model_param.model_name}.{CFG.training_keyword}'

    root_save_dir = '../checkpoint/NER_QA'
    save_dir = os.path.join(root_save_dir, CFG.model_param.model_name)
    os.makedirs(save_dir, exist_ok=True)
    tmp_output_dir = f'{save_dir}/tmp'

    model_path = BERT_MODEL_LIST[CFG.model_param.model_name]

    config = AutoConfig.from_pretrained(model_path)

    if "deberta-v2" in model_path or "deberta-v3" in model_path:
        from transformers.models.deberta_v2 import DebertaV2TokenizerFast
        tokenizer = DebertaV2TokenizerFast.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    data_collator = CustomDataCollator(
        tokenizer,
        pad_to_multiple_of=8 if CFG.train_param.fp16 else None,
        padding=True,
        max_length=int(CFG.data_param.max_seq_length),
    )

    tokenized_dataset = get_tokenized_dataset(tokenizer, CFG)

    training_args = TrainingArguments(
        output_dir=tmp_output_dir,
        do_train=bool(CFG.train_param.do_train),
        do_eval=bool(CFG.train_param.do_eval),
        do_predict=bool(CFG.train_param.do_predict),
        per_device_train_batch_size=int(
            CFG.train_param.per_device_train_batch_size),
        per_device_eval_batch_size=int(
            CFG.train_param.per_device_eval_batch_size),
        gradient_accumulation_steps=int(
            CFG.train_param.gradient_accumulation_steps),
        learning_rate=float(CFG.train_param.learning_rate),
        weight_decay=float(CFG.train_param.weight_decay),
        num_train_epochs=int(CFG.train_param.num_train_epochs),
        lr_scheduler_type=CFG.train_param.lr_scheduler_type,
        warmup_ratio=float(CFG.train_param.warmup_ratio),
        logging_steps=int(CFG.train_param.logging_steps),
        evaluation_strategy=CFG.train_param.evaluation_strategy,
        save_strategy=CFG.train_param.save_strategy,
        seed=int(CFG.train_param.seed),
        fp16=bool(CFG.train_param.fp16),
        report_to="wandb",
        group_by_length=bool(CFG.train_param.group_by_length),
        disable_tqdm=False,
        load_best_model_at_end=True,
    )

    # Initialize our Trainer
    for fold in range(CFG.model_param.n_fold):
        print(f"{yellow_font}====== Fold: {fold} ======{reset_all}")
        model = get_model(model_path, config)

        run = wandb.init(
            project=CFG.program_param.project_name,
            config=CFG,
            job_type='Train',
            group=CFG.group,
            tags=[CFG.model_param.model_name, hash_name],
            name=f'{hash_name}-fold-{fold}',
            anonymous='must'
        )

        trainer = NBMETrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset.filter(
                lambda x: x["fold"] != fold),
            eval_dataset=tokenized_dataset.filter(
                lambda x: x["fold"] == fold),
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        save_path = \
            f"{save_dir}/[{CFG.training_keyword.upper()}]_SCHEDULER_{CFG.train_param.lr_scheduler_type}_FOLD_{fold}"
        trainer.save_model(save_path)
        run.finish()

        shutil.rmtree(tmp_output_dir)
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
