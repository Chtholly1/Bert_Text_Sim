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

    model = NeuralNetwork(model_path, Config, output_way).to(device)

    snil_data_dev = load_snli_vocab(os.path.join(snli_file_path, snli_val_file))
    val_data = generate_treep_data(snil_data_dev)
    val_data = val_data[:val_data_max_num]
    val_dataset = TrainDataset(val_data, tokenizer, maxlen)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    #epochs = 5
    #for t in range(epochs):
    #    print(f"Epoch {t + 1}\n-------------------------------")
    #    train(train_dataloader, test_dataloader, model, optimizer)
    #print("Train_Done!")
    #print("Deving_start!")
    model.load_state_dict(torch.load(load_path))
    #corrcoef = test(deving_data,model)
    corrcoef = test_auc(val_loader,model)
    print(f"dev_corrcoef: {corrcoef:>4f}")
