import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import mojimoji
import re
import jaconv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import warnings
from sklearn import preprocessing
import MeCab
def parse_text(text, debug=False):
    text = re.sub(r'［[^］]+］', ' ', text)  
#     text = re.sub(r'（[^）]+）', ' ', text)  
    text = re.sub(r'○', ' ', text)    
    text = re.sub(r'×', '', text)
    text = re.sub(r'※', '', text)    
#     text = re.sub(r'｜', ' ', text)
    text = re.sub(r'[\s、]', '' , text)
#     text = re.sub(r'一', ' ', text)
#     text = re.sub(r'…', ' ', text)  
#     text = re.sub(r'―', ' ', text)
    text = re.sub(r'[0-9]', '0', text)
    return text

def tokenize1(text):
    available_norm = ['接尾', '一般', '形容動詞語幹', 'サ変接続']
    node = mecab.parseToNode(text)
    l = []
    while node:
        l.append(node.surface)
        node = node.next
    return l
traindf=pd.read_csv("data/train.csv")
testdf=pd.read_csv("data/test.csv")
df=pd.concat([traindf,testdf])
df.body=df.body.map(parse_text)
traindf.body=traindf.body.map(parse_text)
testdf.body=testdf.body.map(parse_text)
y = df['author'].values
mecab = MeCab.Tagger('-Owakati')
count = CountVectorizer(analyzer=tokenize1)
bags = count.fit_transform(df.body.values)

features = count.get_feature_names()

bodyvec = pd.DataFrame(bags.toarray(), columns=features)
wordbody=bodyvec.sum().sort_values(ascending=False)
bodyvec1 = bodyvec[list(wordbody[wordbody > 100].index)]
xsum = bodyvec1.sum(axis=1)

xsum = np.array(xsum).reshape(len(xsum), 1)
newdf = pd.concat([df.reset_index(drop=True), pd.DataFrame(bodyvec1)/xsum*100], axis=1)
train=newdf.dropna().drop(["writing_id","body",],axis=1)
test=newdf[newdf.author.isnull()].drop(["writing_id","body"],axis=1)
test=test.drop(["author"],axis=1)
X = train.drop(["author"],axis=1)
y = train.author

mmscaler = preprocessing.MinMaxScaler()

mmscaler.fit(pd.concat([X,test]))
X = mmscaler.transform(X)
test= mmscaler.transform(test)

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
mlpr=MLPRegressor(**{"hidden_layer_sizes":(128,128,128),"random_state":42})
mlpr.fit(X,y)
pred1 = np.where(mlpr.predict(test)>0.3,1,0)

bodyvec1 = bodyvec[list(wordbody[wordbody > 80].index)]
xsum = bodyvec1.sum(axis=1)

xsum = np.array(xsum).reshape(len(xsum), 1)
newdf = pd.concat([df.reset_index(drop=True), pd.DataFrame(bodyvec1)/xsum*100], axis=1)
train=newdf.dropna().drop(["writing_id","body",],axis=1)
test=newdf[newdf.author.isnull()].drop(["writing_id","body"],axis=1)
test=test.drop(["author"],axis=1)
X = train.drop(["author"],axis=1)
y = train.author

mmscaler = preprocessing.MinMaxScaler()

mmscaler.fit(pd.concat([X,test]))
X = mmscaler.transform(X)
test= mmscaler.transform(test)
mlpr=MLPRegressor(**{"hidden_layer_sizes":(128,128,128),"random_state":42})
mlpr.fit(X,y)
pred2 = np.where(mlpr.predict(test)>0.46,1,0)

bodyvec1 = bodyvec[list(wordbody[wordbody > 37].index)]
xsum = bodyvec1.sum(axis=1)

xsum = np.array(xsum).reshape(len(xsum), 1)
newdf = pd.concat([df.reset_index(drop=True), pd.DataFrame(bodyvec1)/xsum*100], axis=1)
train=newdf.dropna().drop(["writing_id","body",],axis=1)
test=newdf[newdf.author.isnull()].drop(["writing_id","body"],axis=1)
test=test.drop(["author"],axis=1)
X = train.drop(["author"],axis=1)
y = train.author

mmscaler = preprocessing.MinMaxScaler()

mmscaler.fit(pd.concat([X,test]))
X = mmscaler.transform(X)
test= mmscaler.transform(test)
mlpr=MLPRegressor(**{"hidden_layer_sizes":(128,128,128),"random_state":42})
mlpr.fit(X,y)
pred3 = np.where(mlpr.predict(test)>0.33,1,0)

pred=np.where(pred1+pred2+pred3>1.5,1,0)
sub = pd.DataFrame(pd.read_csv("data/test.csv")['writing_id'])
sub["author"] = list(pred)
sub.to_csv("submission.csv", index = False)
