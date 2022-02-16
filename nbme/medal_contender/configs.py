# Config 설정
BERT_MODEL_LIST = {
    "hatebert": "hateBERT",
    "roberta": "roberta-base",
    "roberta-large": "roberta-large",
    "funnel": "intermediate",
    "muppet": "muppet-roberta-base",
    "distilbert": "distilbert-base-cased",
    "electra": "electra-base-discriminator",
    "luke": "luke-base",
    "deberta-v3": "deberta-v3-base",
    "deberta": "microsoft/deberta-base",
    "bigbird_roberta": "bigbird-roberta-base",
    "t5": "t5-base",
    "bert": "bert-base-cased",
    "toxicbert": "toxic-bert",
    "toxic-roberta": "unbiased-toxic-roberta",
    "mpnet": "all-mpnet-base-v2",
    "distilbart": "distilbart-cnn-6-6-large"
}

SCHEDULER_LIST = {
    "cos_ann": 'CosineAnnealingLR',
    "cos_ann_warm": 'CosineAnnealingWarmRestarts',
    "lambda":"LambdaLR",
    "multiple":"MultiplicativeLR",
    "step":"StepLR",
    "mul-step":"MultiStepLR",
    "exp":"ExponentialLR",
    "rlrp":"ReduceLROnPlateau",
    "clr":"CyclicLR",
    "one-clr":"OneCycleLR",
    "none": 'None',
}

OPTIMIZER_LIST = {
    'adamw': 'AdamW',
    'bert': 'BertAdam',
    'lamb': 'LAMB'
}
