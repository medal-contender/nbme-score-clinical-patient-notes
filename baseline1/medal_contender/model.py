import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from medal_contender.configs import SCHEDULER_LIST, BERT_MODEL_LIST
import math

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

class CosineAnnealingWarmupRestarts(_LRScheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr)
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps)
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            

class CosineAnnealingWarmUpRestartsCustom(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestartsCustom, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
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
        model_name = BERT_MODEL_LIST[self.cfg.model_param.model_name]
        if config_path is None:
            self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
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
        
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'CosineAnnealingWarmupRestarts':
        scheduler = CosineAnnealingWarmupRestarts(optimizer = optimizer,
                 first_cycle_steps = num_train_steps, cycle_mult = 1,
                 max_lr = cfg.max_lr, min_lr = cfg.min_lr, warmup_steps = num_train_steps * 0.2,
                 gamma = 1.,last_epoch = -1
        )
        
    elif SCHEDULER_LIST[cfg.model_param.scheduler] == 'CosineAnnealingWarmUpRestartsCustom':
        scheduler = CosineAnnealingWarmUpRestartsCustom(
            optimizer,
            T_0=cfg.train_param.T_0,
            T_mult=cfg.train_param.T_multi, 
            eta_max=float(cfg.train_param.max_lr),
            T_up=cfg.train_param.T_up,
            gamma=float(cfg.train_param.gamma)
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
