# Config 설정
BERT_MODEL_LIST = {
    "deberta": "microsoft/deberta-v3-large",

}

MAKE_TOKENIZER= {
    "transformers": "/home/ubuntu/anaconda3/envs/nbme/lib/python3.7/site-packages/transformers",
    "dir": "./medal_contender/deberta-v2-3-fast-tokenizer"
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
    "cosine":"cosine",
    "none": 'None',
}

OPTIMIZER_LIST = {
    'adamw': 'AdamW',
    'bert': 'BertAdam',
    'lamb': 'LAMB'
}
