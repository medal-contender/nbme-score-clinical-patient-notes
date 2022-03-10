# Config 설정
BERT_MODEL_LIST = {
    "deberta": "microsoft/deberta-base",
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "roberta": "roberta-base",
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
