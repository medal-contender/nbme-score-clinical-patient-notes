import os
import gc
import random

# For text manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
from torch.utils.data import Dataset, DataLoader

# For Transformer Models
from transformers import AutoTokenizer

# Utils
from tqdm import tqdm
from glob import glob
import sys
sys.path.insert(0, '../input/attention3/jigsaw-rate-severity-of-toxic-comments-feature-attention3/jigsaw_toxic_severity_rating/')

from tqdm import tqdm

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# pt파일 경로
MODEL_WEIGHTS = glob('../input/attention3/attention3/*.pt')
MODEL_DIR = '../input/attention3/deberta-v3-base'

CONFIG = dict(
    seed = 42,
    test_batch_size = 128,
    max_length = 128,
    device = torch.device("cuda"),
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
)

class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.text = df['text'].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
                        text,
                        truncation=True,
                        add_special_tokens=True,
                        max_length=self.max_len,
                        padding='max_length'
                    )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']        
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long)
        }    

def set_seed(seed = 42):
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    
@torch.no_grad()
def valid_fn(model, textloader, device):
    model.eval()
    
    PREDS = []
    
    bar = tqdm(enumerate(textloader), total=len(textloader))
    for _, text in bar:
        ids = text['ids'].to(device, dtype = torch.long)
        mask = text['mask'].to(device, dtype = torch.long)
        
        outputs = model(ids, mask)
        PREDS.append(outputs.view(-1).cpu().detach().numpy()) 
    
    PREDS = np.concatenate(PREDS)
    gc.collect()
    
    return PREDS


def inference(model_paths, textloader, device):
    final_preds = []
    for i, path in enumerate(model_paths):

        model = torch.load(path)
        model.to(CONFIG['device'])
        
        print(f"Getting predictions for model {i+1}")
        preds = valid_fn(model, textloader, device)

        final_preds.append(preds)
    
        del model
        _ = gc.collect()
    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds


set_seed(CONFIG['seed'])
df = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")

test_dataset = JigsawDataset(
                    df, 
                    CONFIG['tokenizer'], 
                    max_length=CONFIG['max_length']
                )
test_loader = DataLoader(
                    test_dataset, 
                    batch_size=CONFIG['test_batch_size'],
                    num_workers=2, 
                    shuffle=False, 
                    pin_memory=True
                )

preds = inference(MODEL_WEIGHTS, test_loader, CONFIG['device'])


preds = (preds-preds.min())/(preds.max()-preds.min())


sub_df = pd.DataFrame()
sub_df['comment_id'] = df['comment_id']
sub_df['score'] = preds
sub_df['score'] = sub_df['score'].rank(method='first')
sub_df[['comment_id', 'score']].to_csv("submission.csv", index=False)
