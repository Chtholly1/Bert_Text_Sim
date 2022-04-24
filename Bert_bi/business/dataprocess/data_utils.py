# -*- coding: utf-8 -*-
import os
import time
import json
import random

import numpy as np
import pandas as pd
import collections
from hanziconv import HanziConv
import torch as t
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from conf.config import args

MODEl_NAME='./base_albert'
MAX_LEN = args.max_length
neg_sample_times = 1

def content_process(content, tokenizer):
    #content = str_q2b(content)
    content = HanziConv.toSimplified(content)
    content = tokenizer.tokenize(content)
    return content

class MyDataSet(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        #self.max_seq_len = MAX_LEN
        self.maxlen = MAX_LEN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = self.data[idx]
        sample = self.tokenizer([source[0],source[1],source[2]],max_length=self.maxlen,truncation=True,padding='max_length',return_tensors='pt')
        return sample

def load_data(file_path):
    data = []
    d = dict()
    res_list = []
    with open(file_path) as f:
        for line in f:
            info = json.loads(line.strip())
            sent1, sent2, label = info['sentence1'], info['sentence2'], info['gold_label']
            if not d.get(sent1):
                d[sent1] = ['','']
            if label == 'entailment':
                d[sent1][0] = sent2
            elif label == 'contradiction':
                d[sent1][1] = sent2
    for key in d:
        if key and d[key][0] and d[key][1]:
            res_list.append([key, d[key][0], d[key][1]])
    return res_list

def generate_all_stand(file_path):
    stand_dict = collections.defaultdict(set)
    with open(file_path) as f:
        for line in f:
            talk, stand, label  = line.strip().split('\t')
            stand_dict[label].add(stand)
    stand_dict.pop('other')
    stand_dict.pop('AABB')
    for key in stand_dict:
        stand_dict[key] = list(stand_dict[key])
    return stand_dict

def generate_all_data(file_path, stand_dict):
    data_list = []
    talk_label_dict = collections.defaultdict(list)
    with open(file_path) as f:
        for line in f:
            talk, stand, label = line.strip().split('\t')
            talk_label_dict[talk].append(label)

    for talk in talk_label_dict:
        label_list = talk_label_dict[talk]
        for key in stand_dict:
            temp_talk = talk
            if key not in label_list:
                for item in stand_dict[key]:
                    data_list.append([temp_talk, item, '0', key])
            else:
                for item in stand_dict[key]:
                    data_list.append([temp_talk, item, '1', key])
    return data_list

def generate_data(file_path, stand_dict):
    data_list = []
    idx_map_stand = dict()
    idx = 0
    for key in stand_dict:
        idx_map_stand[idx] = key
        idx += 1
    pos_sample = 0
    neg_sample = 0
    talk_label_dict = collections.defaultdict(list)
    talk_stand_dict = collections.defaultdict(list)
    with open(file_path) as f:
        for line in f:
            talk, stand, label = line.strip().split('\t')
            talk_label_dict[talk].append(label)
            talk_stand_dict[talk].append(stand)
    for talk in talk_label_dict:
        label_list = talk_label_dict[talk]
        if 'other' in label_list or 'AABB' in label_list:
            data_list.append([talk, talk_stand_dict[talk][0], '0', label])
            neg_sample += 1
            continue
        for i,label in enumerate(label_list):
            stand = talk_stand_dict[talk][i]
            if label in stand_dict:
                data_list.append([talk, stand, '1', label])
                pos_sample += 1
            for i in range(neg_sample_times):
                key = ''
                if label in PRICE_LABEL_LIST and random.random() <0.7:
                    while True:
                        i = random.randint(0, len(PRICE_LABEL_LIST)-1)
                        key = PRICE_LABEL_LIST[i]
                        if key not in  label_list:
                            break
                elif label in CONFIG_LABEL_LIST and random.random() < 0.7:
                     while True:
                        i = random.randint(0, len(CONFIG_LABEL_LIST)-1)
                        key = CONFIG_LABEL_LIST[i]
                        if key not in label_list:
                            break 
                elif label in USE_LABEL_LIST and random.random() < 0.3:
                     while True:
                        i = random.randint(0, len(USE_LABEL_LIST)-1)
                        key = USE_LABEL_LIST[i]
                        if key not in label_list:
                            break
                else:
                    while True:
                        i = random.randint(0, idx-1)
                        key = idx_map_stand[i]
                        if key not in label_list:
                            break
                idx_2 = random.randint(0, len(stand_dict[key])-1)
                data_list.append([talk, stand_dict[key][idx_2], '0', label])
                neg_sample += 1
    print(pos_sample, neg_sample)
    return data_list
