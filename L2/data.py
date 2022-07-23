import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json
import logging
import os
from sklearn.model_selection import train_test_split
import collections
import six

class Vocab:
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    return Vocab(vocab_path)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


def cache(func):
    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.split('.')[0] + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f"缓存文件 {data_path} 不存在，重新处理并缓存！")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f"缓存文件 {data_path} 存在，直接载入缓存文件！")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


class LoadDataset:
    def __init__(self,
                 vocab_path='./vocab.txt',  #
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 split_sep='\n',
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True
                 ):
        self.tokenizer = tokenizer
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        # self.UNK_IDX = '[UNK]'

        self.batch_size = batch_size
        self.split_sep = split_sep
        self.max_position_embeddings = max_position_embeddings
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.is_sample_shuffle = is_sample_shuffle

    @cache
    def data_process(self, filepath, postfix='cache'):
        raw_iter = open(filepath).readlines()
        data = []
        max_len = 0
        cnt = 0
        for raw in tqdm(raw_iter, ncols=80):
            info = raw[:-9]
            label = 0 if raw[-9:][0] == 'n' else 1
#            print (raw)

            tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(info)]
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]
            tmp += [self.SEP_IDX]

            tensor_ = torch.tensor(tmp, dtype=torch.long)
            l = torch.tensor(label, dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))
            data.append((tensor_, l))

        return data, max_len

    def load_train_val_test_data(self, train_file_path=None, val_file_path=None, test_file_path=None, only_test=False):
        postfix = str(self.max_sen_len)
        test_data, _ = self.data_process(filepath=test_file_path, postfix=postfix)
        test_iter = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            return test_iter

        train_data, max_sen_len = self.data_process(filepath=train_file_path, postfix=postfix)  # 得到处理好的所有样本
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        val_data, _ = self.data_process(filepath=val_file_path, postfix=postfix)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.generate_batch)
        return train_iter, test_iter, val_iter

    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:  # 对一个batch中的每一个样本进行处理
            batch_sentence.append(sen)
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label
