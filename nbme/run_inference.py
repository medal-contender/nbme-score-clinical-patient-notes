import os
import warnings
import argparse
from glob import glob
from transformers import AutoTokenizer
from medal_challenger.inference import bert_ensemble
from medal_challenger.dataset import prepare_loaders
from medal_challenger.utils import (
    set_seed, get_dataframe, ConfigManager
)
from medal_challenger.configs import BERT_MODEL_LIST

# 경고 억제
warnings.filterwarnings("ignore")

# CUDA가 구체적인 에러를 보고하도록 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main(cfg):

    cfg.tokenizer = AutoTokenizer.from_pretrained(
        f"../models/{BERT_MODEL_LIST[cfg.model_param.model_name]}"
    )

    if "deberta" in cfg.model_param.model_name:
        cfg.infer_param.batch_size = int(cfg.infer_param.batch_size/2)

    set_seed(cfg.program_param.seed)

    # 저장된 경로 
    root_save_dir = '/jigsaw/checkpoint'
    save_dir = os.path.join(root_save_dir,cfg.model_param.model_name)
    assert os.path.isdir(save_dir), f"No Saved Checkpoint"

    os.makedirs(cfg.infer_param.save_dir,exist_ok=True)

    test_csv = "../input/"+cfg.data_param.dir_name.replace("_","-")+f"/{cfg.data_param.infer_file_name}"
    test_df = get_dataframe(test_csv)
    test_loader = prepare_loaders(test_df,cfg,is_train=False)

    MODEL_PATHS = glob(f'{save_dir}/*')

    preds1 = bert_ensemble(MODEL_PATHS, test_loader, cfg)

    preds = (preds1-preds1.min())/(preds1.max()-preds1.min())

    sub_df = get_dataframe(test_csv)
    sub_df['score'] = preds
    sub_df[['comment_id', 'score']].to_csv(f"{cfg.infer_param.save_dir}/submission.csv", index=False)

    print(f'preds.shape : {preds.shape}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file", 
        type=str, 
        required=True,
        help="Type Name Of Config File To Use."
    )
    parser.add_argument(
        "--train", 
        action='store_true',
        help="Toggle On If Model Is On Training."
    )

    args = parser.parse_args()

    cfg = ConfigManager(args).cfg
    main(cfg)