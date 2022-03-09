import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from transformers import AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from medal_contender.configs import SCHEDULER_LIST, BERT_MODEL_LIST

def init_params(module_lst):
    for module in module_lst:
        for param in module.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    return

class NBMEModel(torch.nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False, model_type = 'base'):
        super().__init__()
        self.cfg = cfg
        model_name = f"../models/{BERT_MODEL_LIST[cfg.model_param.model_name]}"
        model_type = 'base' if 'base' in BERT_MODEL_LIST[cfg.model_param.model_name] else 'large'
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                model_name)
            self.config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": 0.2,
                       "layer_norm_eps": 1e-7})
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                model_name, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.dropout = torch.nn.Dropout(p=0.2)

        if model_type == 'large':
            self.classifier = torch.nn.Sequential(
                    nn.Linear(self.config.hidden_size, 512),
                    nn.LayerNorm(512),
                    nn.Dropout(0.2),
                    nn.ReLU(),
                    nn.Linear(512, 1),
                )
        else:
            self.classifier = torch.nn.Sequential(
                    nn.Linear(self.config.hidden_size, 256),
                    nn.LayerNorm(256),
                    nn.Dropout(0.2),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                )
        init_params([self.classifier])
        
    def forward(self, input_ids, attention_mask):
        pooler_outputs = self.backbone(input_ids=input_ids, 
                                       attention_mask=attention_mask)[0]
        logits = self.classifier(self.dropout(pooler_outputs)).squeeze(-1)
        return logits

def fetch_scheduler(optimizer, cfg, num_train_steps=None):
    '''
        Config에 맞는 Solver Scheduler를 반환합니다.
    '''
    if cfg.model_param.scheduler=='linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(cfg.model_param.num_warmup_steps),
            num_training_steps=num_train_steps
        )
    elif cfg.model_param.scheduler=='cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(cfg.model_param.num_warmup_steps),
            num_training_steps=num_train_steps,
            num_cycles=float(cfg.model_param.num_cycles)
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train_param.T_max,
            eta_min=float(cfg.train_param.min_lr)
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.train_param.T_0,
            eta_min=float(cfg.train_param.min_lr)
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'LambdaLR':
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: cfg.train_param.reduce_ratio ** epoch
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'MultiplicativeLR':
        scheduler = lr_scheduler.MultiplicativeLR(
            optimizer,
            lr_lambda=lambda epoch: cfg.train_param.reduce_ratio ** epoch
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.train_param.step_size, gamma=cfg.train_param.gamma
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.train_param.milestones, gamma=cfg.train_param.gamma
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.train_param.gamma
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=2,
            min_lr=float(cfg.train_param.min_lr)
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'CyclicLR':
        scheduler = lr_scheduler.CyclicLR(
            optimizer,
            base_lr=float(cfg.train_param.base_lr),
            step_size_up=cfg.train_param.step_size_up,
            max_lr=float(cfg.train_param.lr),
            gamma=cfg.train_param.gamma,
            mode='exp_range'
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.train_param.max_lr,
            steps_per_epoch=cfg.train_param.steps_per_epoch,
            epochs=cfg.train_param.epochs,
            anneal_strategy='linear'
        )
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'None':
        return None

    return scheduler