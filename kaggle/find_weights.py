import pathlib
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from scipy import stats
from sklearn.metrics import mean_squared_error
from itertools import product
from multiprocessing import freeze_support
import concurrent.futures as cf


data_p = pathlib.Path('D:/kaggle/jigsaw/jigsaw_subs')
sub_df_li = []
for suffix in [908, 910, 907, 911]:
    _df = pd.read_csv(data_p/f'submission{suffix}.csv', index_col=0)
    _df.rename(columns={'score': f'score{suffix}'}, inplace=True)
    sub_df_li.append(_df)
sub_df = pd.concat(sub_df_li, axis=1)
external = sub_df.mean(axis=1).to_frame('score')
external = pd.DataFrame(stats.rankdata(external),
                        columns=['score'], index=sub_df.index)
# external = pd.read_csv(data_p/'submission908.csv', index_col=0)
# model_nm_li = ['roberta', 'electra', 'deberta', 'distilbart', 'bart', 'ridge', 'gpt2', 'muppet', 'roberta_large']
model_nm_li = ['roberta', 'electra', 'deberta', 'distilbart', 'ridge']
# model_nm_li = ['roberta', 'electra', 'deberta', 'distilbart', 'bart', 'ridge']

df_li = []
for model_nm in model_nm_li:
    _df = pd.read_csv(data_p/f'preds_{model_nm}.csv', index_col=0)
    _df.set_index('comment_id', inplace=True)
    df_li.append(_df.rename(columns={'preds': model_nm}))

preds_df = pd.concat(df_li, axis=1)
preds_rank_df = pd.DataFrame(stats.rankdata(preds_df, axis=0),
                             columns=preds_df.columns, index=preds_df.index)


def calc_corr(params):
    scores_li = []
    res_dic = {}
    for i, _nm in enumerate(model_nm_li):
        scores_li.append(preds_rank_df[_nm] * params[i])
        # scores_li.append(preds_rank_df[_nm])
        res_dic[_nm] = params[i]
    pred = np.sum(scores_li, axis=0)
    # pred = np.average(scores_li, weights=params, axis=0)
    pred_rank = stats.rankdata(pred)

    rmse = np.sqrt(mean_squared_error(pred_rank, external))
    res_dic['rmse'] = rmse
    return pd.Series(res_dic)


def run():
    # paramlist = list(product(np.arange(0, 1, 0.1), repeat=len(model_nm_li)))
    paramlist = [
        [i, j, k, l, n]
        for i in np.arange(0.32, 0.65, 0.02)
        for j in np.arange(0.52, 1.0, 0.02)
        for k in np.arange(0.06, 0.25, 0.02)
        for l in np.arange(0.16, 0.45, 0.02)
        for n in np.arange(0.22, 0.50, 0.02)
    ]
    with cf.ProcessPoolExecutor(max_workers=15) as executor:
        res_tup_li = list(
            tqdm(executor.map(calc_corr, paramlist),
                 total=len(paramlist), desc='calculating'))
    # for param in paramlist:
    #     calc_corr(param)
    res_df = pd.concat(res_tup_li, axis=1).T
    res_df.sort_values(by=['rmse'], ascending=True, inplace=True)
    res_df[:100].to_excel('../input/products908+910+907+911v2_rmse_m5_micro.xlsx')


if __name__ == '__main__':
    freeze_support()
    run()
