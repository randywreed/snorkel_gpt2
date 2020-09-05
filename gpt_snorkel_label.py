import os
import yaml
import sys

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython gpt_snorkel_label.py data-dir-path csvfile-dir-path ktrian-dir-path\n'
    )
    sys.exit(1)
data_path= sys.argv[1]
csv_path= sys.argv[2]
ktrain_path=sys.argv[3]

os.makedirs(data_path, exist_ok=True)

#)gpt_snorkel_label.py
#os.path.join(data_path,"/spell/snorkel_fi)les/"
#os.path.join(data_path,'/gdrive/My Drive/)AI & Tech Research/Religion of GPT2/'
filename='biblical_names.csv'

import pandas as pd
df=pd.read_csv(filename,delimiter=',')
Bnames=list(df['terms'])

df.head()

filename='catholic_terms_rev.csv'

import pandas as pd
df=pd.read_csv(filename,delimiter=',')
Cathterms=list(df['terms'])

print(Bnames)

from snorkel.labeling import labeling_function
ABSTAIN=-1
RELIGIOUS=0
SECULAR=1
#FIRST_PER_SG=2
#FIRST_PER_PL=3
#CATHOLICISM=4
#RELIGIONTX=5
#SECULARTX=6

@labeling_function()
def lf_contains_bible_name(x):
  for n in Bnames:
    if str(n).lower() in str(x.text).lower():
      return RELIGIOUS
  return ABSTAIN

@labeling_function()
def lf_contains_bible_quote(x):
  import re
  return RELIGIOUS if re.search(r'\(.+\d*:\d*.+\)',str(x.text)) else ABSTAIN

@labeling_function()
def lf_first_person_sg(x):
  return FIRST_PER_SG if 'I ' in str(x.text) else ABSTAIN

@labeling_function()
def lf_contains_catholicism(x):
  for n in Cathterms:
    if n.lower() in str(x.text).lower():
      return RELIGIOUS
  return ABSTAIN

@labeling_function()
def lf_first_person_pl(x):
  return FIRST_PER_PL if 'we ' in str(x.text).lower() else ABSTAIN

@labeling_function()
def lf_keywords(x):
  keywords=['athiest','athiesm','bible','believe','belief','spiritual','spirtuality','magic']
  return RELIGIOUS if any(word in x.text.lower() for word in keywords) else ABSTAIN

from transformers import pipeline
classifier=pipeline('zero-shot-classification')

@labeling_function()
def lf_0shot_model(x):
  candidate_labels = ["religious", "secular"]
  candidate_results = [0, 0]
  res = classifier(x.text, candidate_labels)
  if res['labels'][0]=='religious' and res['scores'][0]>0.6:
    return RELIGIOUS
  if res['labels'][0]=='secular' and res['scores'][0]>0.6:
    return SECULAR
  return ABSTAIN

import spacy

from snorkel.preprocess import preprocessor
import ktrain
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
from statistics import mean
predictor=ktrain.load_predictor(os.path.join(ktrain_path,'hfdedpredictV3'))
def flmean(x):
    sum=0
    for ele in x:
        sum+=ele
    res=sum/len(x)
    return res


@preprocessor(memoize=True)
def bert_religion_classifier(x):
  sents=sent_tokenize(str(x.text))
  #sentC=[]
  sentR=[]
  sentS=[]
  for sent in sents:
    #sentC.append(predictor.predict(sent))
    p=predictor.predict_proba(str(sent))
    sentR.append(p[0])
    sentS.append(p[1])
  print('sentR={} sentS={}'.format(sentR,sentS))
  if len(sentR)>1:
    R=flmean(sentR)
  else: 
    R=sentR[0]
  if len(sentS)>1:
    S=flmean(sentS)
  else:
    S=sentS[0]
  if R>S:
    x.cat='R'
  else:
    x.cat='S'
  #x.cat=predictor.predict(x.text)
  return x

@labeling_function(pre=[bert_religion_classifier])
def lf_bert_religion(x):
  return RELIGIOUS if x.cat=="R" else ABSTAIN

@labeling_function(pre=[bert_religion_classifier])
def lf_bert_secular(x):
  return SECULAR if x.cat=="S" else ABSTAIN

from snorkel.labeling import LabelingFunction


def keyword_lookup(x, keywords, label):
    try:
      if any(word in x.text.lower() for word in keywords):
          return label
      return ABSTAIN
    except Exception:
      return ABSTAIN

def make_keyword_lf(keywords, label=RELIGIOUS):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )

keyword_bible=make_keyword_lf(keywords=['bible','biblical'])
keyword_belief=make_keyword_lf(keywords=['believe', 'belief'])
keyword_jesus=make_keyword_lf(keywords=['Jesus','christ'])
keyword_atheism=make_keyword_lf(keywords=['atheist','atheism','atheists','atheistic'])
keyword_spiritual=make_keyword_lf(keywords=['spiritual','spirituality'])
keyword_magic=make_keyword_lf(keywords=['magic'])
keyword_mythology=make_keyword_lf(keywords=['zeus','hera','apollo','hermes','myth','mythology','pagan','paganism'])
keyword_creation=make_keyword_lf(keywords=['creator','created','creation'])
from snorkel.labeling import PandasLFApplier
lfs=[lf_contains_bible_name, 
     lf_contains_bible_quote, 
     lf_contains_catholicism,
     lf_bert_religion,
     lf_bert_secular,
     keyword_bible,
     keyword_atheism,
     keyword_belief,
     keyword_jesus,
     keyword_magic,
     keyword_mythology,
     keyword_spiritual,
     keyword_creation,
     lf_0shot_model
     ]

applier=PandasLFApplier(lfs=lfs)

train=pd.read_pickle(os.path.join(csv_path,'train.pkl'))
test=pd.read_pickle(os.path.join(csv_path,'test.pkl'))

Ltrain=applier.apply(df=train)

Ltest=applier.apply(df=test)

#Ltrain[:10]

import numpy as np
Ytest=np.load(os.path.join(csv_path,'Ytest.npy'))
np.save(os.path.join(data_path,'Snorkel_Ltrainv2'),Ltrain)
np.save(os.path.join(data_path,'Snorkel_Ltestv2'),Ltest)
np.save(os.path.join(data_path,'Snorkel_Ytestv2'),Ytest)
