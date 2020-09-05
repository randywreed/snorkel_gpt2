#gpt_snorkel_data_prepare.py
## create/load data
import pandas as pd
import os
import numpy as np
import yaml

os.makedirs('prepare',exist_ok=True)
params=yaml.safe_load(open('params.yaml'))['prepare']
#filename='dedoose_GPT2 Initial prompts and responses.csv'
lines=param['lines']

filename='gpt2HfDedPredictBertv3short.csv'
if not os.path.isfile(filename):
  bfilename="gpt2HfDedPredictBertv3.csv"
  bdf=pd.read_csv(bfilename,error_bad_lines=False)
  if lines>0:
    df=bdf[:lines]
  else:
    df=bdf
else:
  df=pd.read_csv(filename,error_bad_lines=False)
#df.columns=['participant','group','prompt','resp','question','text'] #dedoose gpt2 prompts and responses
df.head()

bad_col=df.columns[0]
df.drop([bad_col],axis=1,inplace=True)
df.rename(columns={'Text':'text'},inplace=True)
train=df.sample(frac=0.8,random_state=200)
test=df.drop(train.index)

train.head()

test.Orig_Cat.value_counts()

test['Orig_Cat']=test['Orig_Cat'].replace(to_replace="R", value=0)
test['Orig_Cat']=test['Orig_Cat'].replace(to_replace="S", value=1)
Ytest=test.Orig_Cat.values

Ytest

print(df.columns[0])
df.head()
train.to_pickle('prepare/train.pkl')
test.to_pickle('prepare/test.pkl')
np.save('prepare/Ytest.npy',Ytest)