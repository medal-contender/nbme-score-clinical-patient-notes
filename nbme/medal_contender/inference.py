import gc
import torch
import numpy as np
from tqdm import tqdm
from medal_contender.model import JigsawModel
from medal_contender.configs import BERT_MODEL_LIST

@torch.no_grad()
def infer_with_one_model(model, dataloader, device):
    model.eval()

    PREDS = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)

        outputs = model(ids, mask)
        PREDS.append(outputs.view(-1).cpu().detach().numpy())

    PREDS = np.concatenate(PREDS)
    gc.collect()

    return PREDS

def bert_ensemble(model_paths, dataloader, cfg):
    final_preds = []
    for i, path in enumerate(model_paths):
        model = JigsawModel(
            f"../models/{BERT_MODEL_LIST[cfg.model_param.model_name]}",
            cfg.model_param.num_classes
        )
        model.to(cfg.model_param.device)
        model.load_state_dict(torch.load(path))

        print(f"Getting predictions for model {i+1}")
        preds = infer_with_one_model(model, dataloader, cfg.model_param.device)
        final_preds.append(preds)

    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds