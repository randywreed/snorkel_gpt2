#gpt_snorkel_label_bert_model.py

import os
import sys
import pandas as pd

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython gpt_snorkel_label_model.py snorkel-model-path bert-model_path\n'
    )
    sys.exit(1)

snorkel_model=sys.argv[1]
bert_model=sys.argv[2]

os.makedirs(snorkel_model,exist_ok=True)
df_train_filtered=pd.read_csv(os.path.join(snorkel_model,'df_train_filtered.csv'))
from ktrain import text
train,val,preprocess=text.texts_from_df(df_train_filtered,'text',['prob_R','prob_S'],ngram_range=2,preprocess_mode='distilbert' )

import ktrain
from ktrain import text
MODEL_NAME = 'distilbert-base-uncased'
#t = text.Transformer(MODEL_NAME, maxlen=500, classes=["R","S"])
#trn = t.preprocess_train(x_train, y_train)
#val = t.preprocess_test(x_test, y_test)
model = preprocess.get_classifier()
learner = ktrain.get_learner(model, train_data=train, val_data=val, batch_size=6)

history=learner.autofit(1e-5,checkpoint_folder='checkpoint',epochs=200)

learner.save_model(os.path.join(bert_model,"distilbert_trained_model"))

predictor = ktrain.get_predictor(learner.model, preproc=preprocess)
predictor.save(os.path.join(bert_model,'predictor'))