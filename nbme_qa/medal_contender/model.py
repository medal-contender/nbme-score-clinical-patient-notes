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
