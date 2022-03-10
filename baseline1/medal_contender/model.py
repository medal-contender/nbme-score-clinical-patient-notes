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

class NBMEModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        model_name = BERT_MODEL_LIST[self.cfg.model_param.model_name]
        model_type = 'base' if 'base' in BERT_MODEL_LIST[cfg.model_param.model_name] else 'large'
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                model_name, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(
                model_name, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.fc_dropout = nn.Dropout(cfg.train_param.fc_dropout)
        # self.fc = nn.Linear(self.config.hidden_size, 1)
        
        if model_type == 'large':
            self.classifier = torch.nn.Sequential(
                    nn.Linear(self.config.hidden_size, self.config.hidden_size),
                    nn.LayerNorm(self.config.hidden_size),
                    nn.Dropout(0.2),
                    nn.ReLU(),
                    nn.Linear(self.config.hidden_size, 1),
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

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.classifier(feature)
        return output

def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters

class DeepShareModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        self.fc_dropout_0 = nn.Dropout(0.1)
        self.fc_dropout_1 = nn.Dropout(0.2)
        self.fc_dropout_2 = nn.Dropout(0.3)
        self.fc_dropout_3 = nn.Dropout(0.4)
        self.fc_dropout_4 = nn.Dropout(0.5)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        output_0 = self.fc(self.fc_dropout_0(feature))
        output_1 = self.fc(self.fc_dropout_1(feature))
        output_2 = self.fc(self.fc_dropout_2(feature))
        output_3 = self.fc(self.fc_dropout_3(feature))
        output_4 = self.fc(self.fc_dropout_4(feature))
        output = (output_0 + output_1 + output_2 + output_3 + output_4) / 5
        return output


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
