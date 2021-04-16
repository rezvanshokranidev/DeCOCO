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
