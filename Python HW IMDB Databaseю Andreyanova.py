#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
from scipy import stats

from pylab import rcParams
rcParams['figure.figsize'] = 12, 6
from matplotlib import pyplot as plt
import zipfile


# In[ ]:


#загрузка файла
z = zipfile.ZipFile("C:/Users/Оля/Documents/Универсл МАГА/HW Python/archive.zip")
df = pd.read_csv(z.open("IMDB Dataset.csv"))
df.info()


# In[ ]:


df.head()


# In[ ]:


# обработка данных
dct = {'positive':1, 'negative':0}
df['sentiment_bin'] = df['sentiment'].map(dct)
df.head()


# In[ ]:


# обработка текста
import nltk
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.tokenize import sent_tokenize, word_tokenize
import re 
 def preproc(sentence):
    sent_text = re.sub(r'[^\w\s]','', sentence)
    words = sent_text.lower().split()
    return(words)
 def senttxt(sent, tokenizer, remove_stopwords=False ):
        raw_sentences = tokenizer.tokenize(oped.strip())
        sentences = []
        for raw_sentence in raw_sentences:
            
            sentences.append(preproc(raw_sentence))
        
        len(sentences)
        return sentences
txt_snt = df['review'].tolist()
sentences = []
for i in range(0,len(nyt_opeds)):
    sent = txt_snt[i].replace("/.", '')
    sentences += senttxt(sent, tokenizer)
    
from gensim.models.word2vec import Word2Vec
model = Word2Vec(size=100, min_count=1)
model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))

class normword2vec():
        
    def transform(self, X, y=None, **fit_params):
        
        X['review'] = X['review'].str.strip()
        X['review'] = X['review'].str.lower()
        X['review'] = X['review'].astype(str)
        X['review'] = [re.sub(r'[^\w\s]', e) for e in X['review']]
     
        return X['review']

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
    
class MeanVect(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(100)], axis=0)
            for words in X
        ])
    
np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(100)], axis=0)

import sklearn
import gensim.sklearn_api
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from gensim.sklearn_api import W2VTransformer

xgb = XGBClassifier()

pipeline = Pipeline([
          ('selectword2vec',  normword2vec()),
      ("word2vec", MeanVect(w2v)),
   
     ('model_fitting',  xgb)]) 
from sklearn import model_selection
from sklearn.model_selection import train_test_split
y = df['y']
X = df
X_train,  X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3,random_state = 2)

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
pd.crosstab(y_test, pred)


# In[ ]:


#1. Наивный Байес.
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools

# для матрицы неточностей
class_names = ["positive", "negative"]
def plot_confusion_matrix(cm, classes, normalize=False, title='Матрица неточностей', cmap=plt.cm.Reds):    
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=0)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.min() + (cm.max() - cm.min()) * 2 / 3.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Истина')
    plt.xlabel('Предсказание')
    plt.tight_layout()

###
train = df[["sentiment_bin"]]
target = df["review"]

gnb = GaussianNB()
model = gnb.fit(train, target)
gprob = gnb.predict_proba(train)
gpred = gnb.predict(train)

acc = accuracy_score(target, gpred)
cnf = confusion_matrix(target, gpred)

print("Точность = %f" % acc)
plot_confusion_matrix(cnf, class_names, normalize=False)


# In[ ]:


from collections import defaultdict
class CategorialNB

def __init__(self, weight = None):
        self.model_0 = [] 
        self.model_1 = []
        
        if weight is None:
            self.weight = [0.5, 0.5] 
        else:
            w = weight[0] + weight[1]
            self.weight = [weight[0] / w, weight[1] / w]
        
    def fit(self, data, target):
        data = np.array(data)
        target = np.array(target)
        
        N = data.shape[0]
        
        feature_count = data.shape[1]

        if data.shape[0] != target.shape[0]:
            raise Exception("Invalid shapes of data vector and target vector")
            
        if np.logical_not(np.isin(target, [0, 1])).any():
            raise Exception("Invalid target vector")

        self.model_1 = [{} for _ in range(feature_count)]
        self.model_0 = [{} for _ in range(feature_count)]
            
        mask_1 = target == 1
        mask_0 = target == 0                    
            
        for i in range(feature_count):
            cats, counts = np.unique(data[mask_1, i], return_counts=True)
            probs = counts / N 
            for category, probability in zip(cats, probs):
                self.model_1[i][category] = probability
                
            cats, counts = np.unique(data[mask_0, i], return_counts=True)
            probs = counts / N             
            for category, probability in zip(cats, probs):
                self.model_0[i][category] = probability
        #print(self.model)
        return self
    
    def predict_proba(self, data)
        data = np.array(data)
        result = np.zeros((data.shape[0], 2))
        result[:, 0] = self.weight[0]
        result[:, 1] = self.weight[1]
        
        for i, row in enumerate(data):
            for j, feature in enumerate(row):
                result[i, 0] = result[i, 0] * self.model_0[j][feature]
                result[i, 1] = result[i, 1] * self.model_1[j][feature]        
        return result / result.sum(axis=1)[:, None]
    
    def predict(self, data):
        proba = self.predict_proba(data)                
        mask = proba[:, 1] > proba[:, 0]
        return mask.astype("int")


# In[ ]:


#2. Дерево решений.
from sklearn.model_selection import train_test_split
data = df[["review", "sentiment"]].values
target = df['sentiment'].values
train, test, target_train, target_test = train_test_split(   
    data, target, 
    test_size=0.3)


# In[ ]:


class RecursiveTree:
    def __init__(self, max_depth): 
        self.max_depth = max_depth

        self.p0 = None
        self.p1 = None
        self.size = None
        self.target = None
        self.entropy = None
        
        self.feature_num = None
        self.feature_value = None
        self.childs = None
        
    def set_depth(self, max_depth):
        self.max_depth = max_depth
        if self.childs is not None:
            self.childs[0].set_depth(max_depth - 1)
            self.childs[1].set_depth(max_depth - 1)
    
    def _entropy(self, values):
        p = values.sum() / values.shape[0]
        q = 1.0 - p        
        return - np.nan_to_num(p * np.log2(p)) - np.nan_to_num(q * np.log2(q))
    
    def print(self, names, tab=0):              
        if self.childs is not None:
            print("  "*tab, "[%4s == %2d][%6d]" % (names[self.feature_num], self.feature_value, self.size), 
              "%4.2f %4.2f %2d %7.5f" %(self.p0, self.p1, self.target, self.entropy))
            self.childs[0].print(names, tab+1)
            self.childs[1].print(names, tab+1)
        else:
            print("  "*tab, "[____ == __][%6d]" % self.size, 
              "%4.2f %4.2f %2d %7.5f" %(self.p0, self.p1, self.target, self.entropy))
            
    def _get_dot_code(self, names, name, parent=None):
        content = "\n"

        if self.childs is not None:
            content += '%s [label="%s == %s\\nS = %.3f\\nsamples = %d\\nprob = [%.2f, %.2f]", fillcolor="#%X"];\n' % (
                name, 
                names[self.feature_num], self.feature_value, 
                self.entropy, self.size, 
                self.p0, self.p1,
                (0xe5813900 if self.p0 > self.p1 else 0x399de500) 
                + int(0xff * (self.p0 if self.p0 > self.p1 else self.p1))
            )               
        else:
            content += '%s [label="S = %.3f\\nsamples = %d\\nprob = [%.2f, %.2f]", fillcolor="#%X"];\n' % (
                name, 
                self.entropy, self.size, 
                self.p0, self.p1,
                (0xe5813900 if self.p0 > self.p1 else 0x399de500) 
                + int(0xff * (self.p0 if self.p0 > self.p1 else self.p1))
            )               
            
        if parent is not None:
            content += "%s -> %s;" % (parent, name)
            
        if self.childs is not None:
            content += self.childs[0]._get_dot_code(names, name + "f", name)
            content += self.childs[1]._get_dot_code(names, name + "t", name)
            
        return content
            
    def to_dot(self, filename, names):
        f = open(filename, "w")
        content = self._get_dot_code(names, "root", None)
        f.write("digraph Tree {\n")
        f.write('\tnode [shape=box, style="filled", color="black"];\n')
        f.write(content)
        f.write("}")
        f.close()        
        
    def _predict_proba(self, features):
        if self.childs is None or self.max_depth <= 0:
            return self.p0, self.p1
        
        if features[self.feature_num] == self.feature_value:
            return self.childs[1]._predict_proba(features)
        else:
            return self.childs[0]._predict_proba(features)
        
    def _predict(self, features):
        if self.childs is None or self.max_depth <= 0:
            return self.target
        
        if features[self.feature_num] == self.feature_value:
            return self.childs[1]._predict(features)
        else:
            return self.childs[0]._predict(features)
        
    def predict(self, data):
        data = np.array(data)
        result = np.zeros(data.shape[0])
        for i, features in enumerate(data):
            result[i] = self._predict(features)
        return result
    
    def predict_proba(self, data):
        data = np.array(data)
        result = np.zeros( (data.shape[0], 2) )
        for i, features in enumerate(data):
            result[i] = self._predict_proba(features)
        return result
    
    def fit(self, data, target):
        data = np.array(data)
        target = np.array(target)
    
        mask = target == 0
        self.size = target.shape[0]
        self.p0 = mask.sum() / self.size
        self.p1 = 1.0 - self.p0
        self.target = 1 if self.p1 > self.p0 else 0
        self.entropy = self._entropy(target)        
        
        if self.entropy == 0:
            return
                     
        n_features = data.shape[1]        
        features = [np.unique(data[:,i]) for i in range(n_features)]
        
        split_best = None
        split_feature = None
        split_feature_value = None
        split_mask_true = None
        split_mask_false = None
        
        for i, feature in enumerate(features):
            if len(feature) < 2:
                continue
                
            for fv in feature:
                mask = data[:, i] == fv
                not_mask = np.logical_not(mask)
                
                S_true = self._entropy(target[mask])
                S_false = self._entropy(target[not_mask])
                
                p_true = mask.sum() / mask.shape[0]
                p_false = 1.0 - p_true
                
                dS = p_true * S_true + p_false * S_false
                
                if split_best is None or split_best > dS:
                    split_best = dS
                    split_feature = i
                    split_feature_value = fv
                    split_mask_true = mask
                    split_mask_false = not_mask

        if split_best is None:
            return
                   
        self.feature_num = split_feature
        self.feature_value = split_feature_value

        self.childs = [RecursiveTree(self.max_depth - 1), RecursiveTree(self.max_depth - 1)]
        self.childs[0].fit(data[split_mask_false], target[split_mask_false])
        self.childs[1].fit(data[split_mask_true], target[split_mask_true])


# In[ ]:


from sklearn.metrics import accuracy_score

tree = RecursiveTree(3)
tree.fit(train, target_train)
pv = tree.predict_proba(test)
yv = tree.predict(test)

print("Accuracy = ", accuracy_score(target_test, yv))

tree.to_dot("graph.dot", names=["review", "sentiment"])


# In[ ]:


get_ipython().system('dot -Tpng "graph.dot" -o "graph.png"')


# In[ ]:


#3. Метод k - ближайших соседей


# In[ ]:


#4. Линейные модели

