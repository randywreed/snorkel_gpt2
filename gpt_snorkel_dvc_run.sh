#!/bin/sh
dvc run -n prepare -p -d gpt_snorkel_data_prepare.py -o prepare python3 gpt_snorkel_data_prepare.py
dvc run -n label -d prepare -d gpt_snorkel_label.py -o label python3 gpt_snorkel_label.py label prepare ktrain
dvc run -n label_model -d label -d gpt_snorkel_label_model.py -o snorkel_model python3 gpt_snorkel_label_model.py snorkel_model label prepare
dvc run -n bert_model -d snorkel_model -d gpt_snorkel_label_bert_model.py -o bert_model python3 gpt_snorkel_label_bert_model snorkel_model bert_model