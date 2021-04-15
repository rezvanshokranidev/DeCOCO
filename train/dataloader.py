import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import gc
from pathlib import Path as path


dir = path.cwd()


class IEMOCAPRobertaCometDataset(Dataset):
    def __init__(self, split):
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

        self.speakers, self.labels, \
        self.roberta,_,_,_,\
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open(dir / 'train/erc-training/iemocap/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open(dir / 'train/erc-training/iemocap/iemocap_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta[vid]),\
               torch.FloatTensor(self.xIntent[vid]),\
               torch.FloatTensor(self.xAttr[vid]),\
               torch.FloatTensor(self.xNeed[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xEffect[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oEffect[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.speakers[vid]]),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(
        self, data
    ):
        '''
            collate_fn
            the default collate_fn simply converts NumPy arrays in PyTorch tensors. 
            When automatic batching is enabled, collate_fn is called with a list 
            of data samples at each time. It is expected to collate the input samples 
            into a batch for yielding from the data loader iterator.
        '''
        data = pd.DataFrame(data)
        # print('***********************')
        # a=  [data[i] for i in data]
        # print(a)
        # sys.exit()
        '''
            pad_sequence
            pad_sequence stacks a list of Tensors along a new dimension, 
            and pads them to equal length. For example, 
            if the input is list of sequences with size L x * and 
            if batch_first is False, and T x B x * otherwise.

            B is batch size. 
            It is equal to the number of elements in sequences. 
            T is length of the longest sequence. 
            L is length of the sequence. 
            * is any number of trailing dimensions, including none.
            
            This function returns a Tensor of size T x B x * or B x T x * 
            where T is the length of the longest sequence. ... 
            This function assumes trailing dimensions and type of all the Tensors in sequences are same.
        '''
        return [
            pad_sequence(data[i]) if i < 11 else
            pad_sequence(data[i], True) if i < 13 else data[i].tolist()
            for i in data
        ]


class MELDRobertaCometDataset(Dataset):
    def __init__(self, split):
        '''
        label index mapping =
        '''
        self.speakers, self.emotion_labels, self.labels, \
        self.roberta,_,_,_,\
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open(dir / 'train/erc-training/meld/meld_features_roberta.pkl', 'rb'), encoding='latin1')

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open(dir / 'train/erc-training/meld/meld_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta[vid]),\
               torch.FloatTensor(self.xIntent[vid]),\
               torch.FloatTensor(self.xAttr[vid]),\
               torch.FloatTensor(self.xNeed[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xEffect[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oEffect[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor(self.speakers[vid]),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        data = pd.DataFrame(data)
        return [
            pad_sequence(data[i]) if i < 11 else
            pad_sequence(data[i], True) if i < 13 else data[i].tolist()
            for i in data
        ]


class DailyDialogueRobertaCometDataset(Dataset):
    def __init__(self, split):
        torch.cuda.empty_cache()
        gc.collect()

        self.speakers, self.labels, \
        self.roberta,_,_,_,\
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open(dir / 'train/erc-training/dailydialog/dailydialog_features_roberta.pkl', 'rb'), encoding='latin1')

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open(dir / 'train/erc-training/dailydialog/dailydialog_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta[vid]),\
               torch.FloatTensor(self.xIntent[vid]),\
               torch.FloatTensor(self.xAttr[vid]),\
               torch.FloatTensor(self.xNeed[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xEffect[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oEffect[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.speakers[vid]]),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        data = pd.DataFrame(data)
        return [
            pad_sequence(data[i]) if i < 11 else
            pad_sequence(data[i], True) if i < 13 else data[i].tolist()
            for i in data
        ]


class EmoryNLPRobertaCometDataset(Dataset):
    def __init__(self, split):
        '''
        label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
        '''

        self.speakers, self.labels, \
        self.roberta,_,_,_,\
        self.sentences, self.trainId, self.testId, self.validId \
        = pickle.load(open(dir / 'train/erc-training/emorynlp/emorynlp_features_roberta.pkl', 'rb'), encoding='latin1')


        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open(dir / 'train/erc-training/emorynlp/emorynlp_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta[vid]),\
               torch.FloatTensor(self.xIntent[vid]),\
               torch.FloatTensor(self.xAttr[vid]),\
               torch.FloatTensor(self.xNeed[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xEffect[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oEffect[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.speakers[vid]]),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        data = pd.DataFrame(data)
        return [
            pad_sequence(data[i]) if i < 11 else
            pad_sequence(data[i], True) if i < 13 else data[i].tolist()
            for i in data
        ]
