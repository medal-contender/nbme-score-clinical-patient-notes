import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
stop = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in text]

def clean(data):
    
    data = data.replace(r"what's", "what is ")    
    data = data.replace(r"\'ve", " have ")
    data = data.replace(r"can't", "cannot ")
    data = data.replace(r"n't", " not ")
    data = data.replace(r"i'm", "i am ")
    data = data.replace(r"\'re", " are ")
    data = data.replace(r"\'d", " would ")
    data = data.replace(r"\'ll", " will ")
    data = data.replace(r"\'scuse", " excuse ")
    data = data.replace(r"\'s", " ")
    data = data.replace(r"@USER", "")
    
    # Clean some punctutations
    data = data.replace('\n', ' \n ')
    data = data.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3')
    # Replace repeating characters more than 3 times to length of 3
    data = data.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1')    
    # Add space around repeating characters
    data = data.replace(r'([*!?\']+)',r' \1 ')    
    # patterns with repeating characters 
    data = data.replace(r'([a-zA-Z])\1{2,}\b',r'\1\1')
    data = data.replace(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1')
    data = data.replace(r'[ ]{2,}',' ').strip()   
    data = data.replace(r'[ ]{2,}',' ').strip()   
    
    return data


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




df['comment_text'] = df['comment_text'].apply(lambda x: clean(x))
df['comment_text'] = df['comment_text'].apply(lambda x: text_cleaning(x))
df['comment_text'] = df['comment_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df = df.reset_index(drop=True)
df.rename(columns = {'index' : 'comment_id','comment_text' : 'text'}, inplace = True)
