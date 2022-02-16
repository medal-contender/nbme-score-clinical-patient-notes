import re
import numpy as np
import pandas as pd
from scipy import sparse
from bs4 import BeautifulSoup
from sklearn.linear_model import Ridge
from gensim.models import KeyedVectors, FastText
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings; warnings.filterwarnings("ignore")

N_MODELS = 4
EXTRA_DIM = 256
ALPHA_STEP_SIZE = 0.5


def text_cleaning(text):
    template = re.compile(r'https?://\S+|www\.\S+')
    text = template.sub(r'', text)
    soup = BeautifulSoup(text, 'lxml')
    only_text = soup.get_text()
    text = only_text
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags = re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r"[^a-zA-Z\d]", " ", text)
    text = re.sub(' +', ' ', text)
    text = text.strip()
    return text

df = pd.read_csv('../input/jigsaw-regression-based-data/train_data_version2.csv')
df = df.dropna(axis = 0)
vec = TfidfVectorizer(min_df = 3, max_df = 0.5, analyzer = 'char_wb', ngram_range = (3, 5), max_features = 46000)
vec.fit(df['text'])
fmodel = FastText.load('../input/jigsaw-regression-based-data/FastText-jigsaw-256D/Jigsaw-Fasttext-Word-Embeddings-256D.bin')

def splitter(text): return [word for word in text.split(' ')]
def vectorizer(text):
    tokens = splitter(text)
    x1 = vec.transform([text]).toarray()
    x2 = np.mean(fmodel.wv[tokens], axis = 0).reshape(1, -1)
    x = np.concatenate([x1, x2], axis = -1).astype(np.float16)
    del x1
    del x2
    return x

X_np = np.array([vectorizer(text) for text in df.text]).reshape(-1, (len(vec.vocabulary_) + EXTRA_DIM))
X = sparse.csr_matrix(X_np)
del X_np

class RidgeEnsemble():
    def __init__(self, n_models = 4, alpha_step_size = 0.5): self.models = [Ridge(alpha = alpha) for alpha in [alpha_step_size * i for i in range(1, n_models + 1)]]
    def fit(self, X, y): self.models = [model.fit(X, y) for model in self.models]
    def predict(self, X): return np.mean(np.concatenate([np.expand_dims(model.predict(X), axis = 0) for model in self.models], axis = 0), axis = 0)

model = RidgeEnsemble()
model.fit(X, df['y'])

df_val = pd.read_csv("../input/jigsaw-toxic-severity-rating/validation_data.csv")

X_less_toxic_temp = []
for text in df_val.less_toxic: X_less_toxic_temp.append(vectorizer(text))
X_less_toxic_temp = np.array(X_less_toxic_temp).reshape(-1, (len(vec.vocabulary_) + EXTRA_DIM))
X_less_toxic = sparse.csr_matrix(X_less_toxic_temp)
del X_less_toxic_temp

X_more_toxic_temp = []
for text in df_val.more_toxic: X_more_toxic_temp.append(vectorizer(text))
X_more_toxic_temp = np.array(X_more_toxic_temp).reshape(-1, (len(vec.vocabulary_) + EXTRA_DIM))
X_more_toxic = sparse.csr_matrix(X_more_toxic_temp)
del X_more_toxic_temp

df_sub2 = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")
df_sub2['text'] = df_sub2['text'].apply(text_cleaning)
X_sub_temp = []
for text in df_sub2.text: X_sub_temp.append(vectorizer(text))
X_sub_temp = np.array(X_sub_temp).reshape(-1, (len(vec.vocabulary_) + 256))
X_test = sparse.csr_matrix(X_sub_temp)
del X_sub_temp

preds4 = model.predict(X_test)
preds4 = (preds4-preds4.min())/(preds4.max()-preds4.min())

df_sub2['score'] = preds4
df_sub2[['comment_id', 'score']].to_csv("submission.csv", index=False)
