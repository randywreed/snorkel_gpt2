#gpt_snorkel_label_model.py

import numpy as np
import os
import yaml
import sys
import pandas as pd

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython gpt_snorkel_label_model.py data-dir-path model-dir-path\n'
    )
    sys.exit(1)


data_path = sys.argv[1]
modelname = sys.argv[2]

Ltrain=np.load(os.path.join(data_path,'Ltrain.npy'))
Ltest=np.load(os.path.join(data_path,'Ltest.npy'))
Ytest=np.load(os.path.join(data_path,'Ytest.npy'))
train=pd.read_pickle('train.pkl')

from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=Ltrain, n_epochs=1000, log_freq=100, seed=123)
prob_train=label_model.predict_proba(L=Ltrain)

from snorkel.labeling.model import MajorityLabelVoter

majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=Ltrain)

majority_acc = majority_model.score(L=Ltest, Y=Ytest, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

label_model_acc = label_model.score(L=Ltest, Y=Ytest, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")

#!ls

label_model.save(os.path.join(modelname,'save_snorkel_label_model.pkl'))
majority_model.save(os.path.join(modelname,'save_snorkel_label_model.pkl'))

from snorkel.labeling import filter_unlabeled_dataframe
df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(X=train, y=prob_train,L=Ltrain)

cnt=0
for a in probs_train_filtered:
  #print(a)
  print(len(a))
  print('{0:.10f}'.format(a[0]))
  print('{0:.10f}'.format(a[1]))
  #print('{0:.10f}'.format(a[2]))
  #print('{0:.10f} {0:.10f} {0:.10f}'.format(a[0],a[1],a[2]))
  cnt+=1
  if cnt>10:
    break
print('df_train_filtered')
print(df_train_filtered.head())
print(df_train_filtered.columns)
#df_train_filtered.drop('prob_cat',axis=1)

#add labels to df
df_train_filtered['prob_R']=probs_train_filtered[:,0]
df_train_filtered['prob_S']=probs_train_filtered[:,1]

def addNewCat(row):
  R=row['prob_R']
  S=row['prob_S']
  if R>S:
    return 'R'
  if S>R:
    return 'S'
  return '?'

df_train_filtered['Snokel_Cat']=df_train_filtered.apply(lambda row: addNewCat(row),axis=1)
df_train_filtered.to_csv(os.path.join(data_path,'df_train_filtered'))

df_train_filtered.head()

edf=df_train_filtered.loc[~(df_train_filtered['Cat']==df_train_filtered['Snokel_Cat'])]

edf.shape

edf.to_csv(os.path.join(data_path,'snorkel_errors.csv'))