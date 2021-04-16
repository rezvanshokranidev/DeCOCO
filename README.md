# DeCOCO
## **DEtecting emotion shift in COnversations by COmmonsense knowledge**
To set up project go through the following stepd:
- Extract Features by using RoBERTa and COMET. Due to complicated steps to run RoBERTa and COMET you can download the result of step 1-6 from [here](https://drive.google.com/file/d/1TQYQYCoPtdXN2rQ1mR2jisjUztmOzfZr/view)
  1. You can follow step 1-3 or using tutorial of [here](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md)
  2. Preprocess the data: Data should be preprocessed following the language modeling format and use GPT-2 BPE (Byte Pair Encoding) tokenizer.
  conv-emotion\COSMIC\feature-extraction\roberta_preprocess_iemocap.sh
  3. Train RoBERTa base: conv-emotion\COSMIC\feature-extraction\roberta_preprocess_iemocap.sh
  conv-emotion\COSMIC\feature-extraction\roberta_train_iemocap.sh
  4. Load your pretrained model: conv-emotion\COSMIC\feature-extraction\roberta_feature_extract_iemocap.py
  5. Clone COMET project from [here](https://github.com/atcbosselut/comet-commonsense)
  6. Download pretrained atomic model from [here](https://drive.google.com/file/d/1vNi4TViLKX_V_wGVXfhpvKimqMjhGBNX/view?usp=sharing) to extract common-sense features.
- Keep the result of prior step in train/erc-training
- To run the project, install prerequisites
  1. python                    3.8.8
  2. torch                     1.8.1
  3. scikit-learn              0.24.1
  4. plotly                    4.14.3
  5. pandas                    1.2.3
  6. numpy                     1.19.5
  7. notebook                  6.3.0
  8. ipython                   7.21.0
- Training, validation, testing, and evaluation of each dataset, have be done by the followin steps:
  - python train_iemocap.py
  - python train_meld.py
  - python train_dailydialog.py
  - python train_emorynlp.py
- You can change the default value of hyperparameter and argument. There are different hyperparameters and argumens you can change by the following stepa:
  - no-cuda: Run model on GPU or CPU
  - lr: Learning rate
  - l2: L2 regularization weight
  - recurrent_dropout: recurrent_dropout of GRU
  - dropout: Dropout rate for hidden layer
  - batch-size
  - epochs
  - class-weight: Use class weight form imbalanced dataset
  - active-listener: This is Monologue and Dialogue
  - attention: Attention type in context GRU
  - seed
  - norm: normalization strategy
- You can pass the value to each dataset like this: python train_iemocap.py --no-cuda --epochs=50 --class-weight --active-listener
- The result will be saved in train/logs
