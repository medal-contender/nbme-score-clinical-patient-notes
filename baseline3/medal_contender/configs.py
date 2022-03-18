# Config 설정
BERT_MODEL_LIST = {
    "deberta": "microsoft/deberta-base",
    "deberta-v3": "microsoft/deberta-v3-base",
    "electra": "../models/electra-base-discriminator",
    "roberta": "../models/roberta-base",
    "roberta-large": "../models/roberta-large",
    "deberta-large": "microsoft/deberta-large",
    "electra-large": "google/electra-large-discriminator",
    "pubmedbert-base": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
}

SCHEDULER_LIST = {
    "cos_ann": 'CosineAnnealingLR',
    "cos_ann_warm": 'CosineAnnealingWarmRestarts',
    "lambda": "LambdaLR",
    "multiple": "MultiplicativeLR",
    "step": "StepLR",
    "mul-step": "MultiStepLR",
    "exp": "ExponentialLR",
    "rlrp": "ReduceLROnPlateau",
    "clr": "CyclicLR",
    "one-clr": "OneCycleLR",
    "cosine": "cosine",
    "none": 'None',
}

OPTIMIZER_LIST = {
    'adamw': 'AdamW',
    'bert': 'BertAdam',
    'lamb': 'LAMB'
}
