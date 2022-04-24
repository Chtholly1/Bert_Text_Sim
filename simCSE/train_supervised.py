#!/usr/bin/env python
# encoding: utf-8

import time
import json
import numpy as np
import torch
import random
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import BertTokenizerFast, AlbertModel, AlbertConfig
import scipy.stats
from tqdm import tqdm
import os

from bussiness.data_process.data_utils import load_snli_vocab, load_STS_data, generate_treep_data, TrainDataset, TestDataset
from bussiness.tools.tools import setup_seed
from bussiness.models.Bert import NeuralNetwork
from bussiness.model_plant import compute_corrcoef, compute_loss, test_acc, test_auc, test, train
from conf.config import *

if __name__ == '__main__':
    print("Using {} device".format(device))
    setup_seed(2000)

    snil_vocab = load_snli_vocab(os.path.join(snli_file_path,snli_train_file))
    #snil_vocab.sort(key=lambda x:x[0])
    snil_treep = generate_treep_data(snil_vocab)
    random.shuffle(snil_treep)
    print(len(snil_treep))

    simCSE_data = snil_treep[:train_data_max_num]
    test_data = snil_treep[train_data_max_num:train_data_max_num+val_data_max_num]

    training_data = TrainDataset(simCSE_data,tokenizer,maxlen)
    train_dataloader = DataLoader(training_data,batch_size=batch_size)

    testing_data = TrainDataset(test_data,tokenizer,maxlen)
    test_dataloader = DataLoader(testing_data,batch_size=batch_size)

    model = NeuralNetwork(model_path, Config, output_way).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, test_dataloader, model, optimizer)
    print("Train_Done!")
    print("Deving_start!")
    model.load_state_dict(torch.load(save_path))
    corrcoef = test_auc(test_dataloader,model)
    print(f"dev_corrcoef: {corrcoef:>4f}")
