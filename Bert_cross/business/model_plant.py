# -*- coding: utf-8 -*-
import os
import time
import random

import numpy as np
import pandas as pd
import torch
import torch as t
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from conf.config import args

def correct_predictions(output_probabilities, targets):
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()

def my_round(prob, ths):
    if prob > ths:
        return 1
    else:
        return 0

def calc_every_acc(pred_dict, probabilities, labels):
    _, out_classes = probabilities.max(dim=1)
    for idx, label in enumerate(labels):
        if label == out_classes[idx]:
            pred_dict[label][1] += 1
        pred_dict[label][0] += 1
    return

def calc_every_label_acc(data_list, y_true, y_pred):
    label_dict = dict()
    label_all_dict = dict()
    for item in data_list:
        label_dict[item[-1]] = [0, 0]
        label_all_dict[item[-1]] = [[], []]
    label_all_dict['all'] = [[], []]
    label_dict['all'] = [0, 0]
    for item, y, yp in zip(data_list, y_true, y_pred):
        label_name = item[-1]
        label_all_dict[label_name][0].append(y)
        label_all_dict[label_name][1].append(my_round(yp, 0.95))
        label_all_dict['all'][0].append(y)
        label_all_dict['all'][1].append(my_round(yp, 0.95)) 
        label_dict[label_name][0] += 1
        label_dict['all'][0] += 1
        if yp > 0.5 and y == 1:
            label_dict[label_name][1] += 1
            label_dict['all'][1] += 1
        elif yp<0.5 and y == 0:
            label_dict[label_name][1] += 1
            label_dict['all'][1] += 1
    return label_dict, label_all_dict
    
def save_result(data_list, y_true, y_pred, output_file='./result.csv'):
    data = []
    f = open(output_file, 'w', encoding='utf-8')
    for item, y, y_ in zip(data_list, y_true, y_pred):
        text = item[1]
        f.write("%s\t%s\t%s\n"%(text, y, y_))
    f.close()
    return
            
def validate(model, dataloader, device, ema=None, output_file=None):
    # Switch to evaluate mode.
    model.eval()
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    tid1s = []
    tid2s = []
    if ema:
        ema.apply_shadow()
    with t.no_grad():
        for input_ids_p, att_mask_p, token_type_ids_p, labels_p, input_ids_n, att_mask_n, token_type_ids_n, labels_n in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if it is used.
            input_ids = torch.cat((input_ids_p, input_ids_n),0).to(device)
            att_mask = torch.cat((att_mask_p, att_mask_n),0).to(device)
            token_type_ids = torch.cat((token_type_ids_p, token_type_ids_n),0).to(device)
            labels = torch.cat((labels_p, labels_n),0).to(device)

            loss, logits, probabilities = model(input_ids, att_mask, token_type_ids, labels)
            loss = loss.mean()
            running_loss += loss.item()
            running_accuracy += correct_predictions(probabilities, labels)
            all_labels.extend(labels.detach().cpu().tolist())
    if ema:
        ema.restore()
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (2*len(dataloader.dataset))
    #save_result(all_labels, all_prob, output_file=output_file)
    return epoch_time, epoch_loss, epoch_accuracy

def test(model, dataloader, device, output_file=None):
    model.eval()
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    tid1s = []
    tid2s = []
    with t.no_grad():
        for input_ids_p, att_mask_p, token_type_ids_p, labels_p, input_ids_n, att_mask_n, token_type_ids_n, labels_n in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if it is used.
            input_ids = torch.cat((input_ids_p, input_ids_n),0).to(device)
            att_mask = torch.cat((att_mask_p, att_mask_n),0).to(device)
            token_type_ids = torch.cat((token_type_ids_p, token_type_ids_n),0).to(device)
            labels = torch.cat((labels_p, labels_n),0).to(device)

            loss, logits, probabilities = model(input_ids, att_mask, token_type_ids, labels)
            loss = loss.mean()
            running_loss += loss.item()
            running_accuracy += correct_predictions(probabilities, labels)
            all_labels.extend(labels.detach().cpu().tolist())
            #_, out_classes = probabilities.max(dim=1)
            all_prob.extend(probabilities[:,1].detach().cpu().tolist())
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / len(dataloader.dataset)
    auc = roc_auc_score(all_labels, all_prob)
    #save_result(data_list, all_labels, all_prob, output_file=output_file)
    return epoch_time, epoch_loss, epoch_accuracy, auc

def train_mixup(model, dataloader, optimizer, max_gradient_norm, device, fgm=None, pgd=None, ema=None):
    #model_bert.eval()
    model.train()
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, (input_ids, att_mask, labels) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        # Move input and output data to the GPU if it is used.
        input_ids = input_ids.to(device)
        att_mask = att_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, logits, probabilities = model(input_ids, att_mask, labels)
        loss.mean()
        if np.isnan(loss.cpu().detach().numpy()):
            continue
        loss.backward()
        if fgm:
            fgm.attack()
            loss_adv, adv_logits, adv_probabilities = model(input_ids, att_mask, labels)
            loss_adv = loss_adv.mean()
            loss_adv.backward()
            fgm.restore()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        if ema:
            ema.update()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probabilities, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1), running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy

def train(model, dataloader, optimizer, max_gradient_norm, device, fgm=None, pgd=None, ema=None):
    #model_bert.eval()
    model.train()
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, (input_ids_p, att_mask_p, token_type_ids_p, labels_p, input_ids_n, att_mask_n, token_type_ids_n, labels_n) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        # Move input and output data to the GPU if it is used.
        input_ids = torch.cat((input_ids_p, input_ids_n),0).to(device)
        att_mask = torch.cat((att_mask_p, att_mask_n),0).to(device)
        token_type_ids = torch.cat((token_type_ids_p, token_type_ids_n),0).to(device)
        labels = torch.cat((labels_p, labels_n),0).to(device)

        optimizer.zero_grad()
        loss, logits, probabilities = model(input_ids, att_mask, token_type_ids, labels)
        if np.isnan(loss.cpu().detach().numpy()):
            continue
        loss = loss.mean()
        loss.backward()
        if fgm:
            fgm.attack()
            loss_adv, adv_logits, adv_probabilities = model(input_ids, att_mask, token_type_ids, labels)
            loss_adv = loss_adv.mean()
            loss_adv.backward()
            fgm.restore()
        if pgd:
            K = 3
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K-1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                loss_adv, _, _ = model(input_ids, att_mask, labels)
                loss_adv = loss_adv.mean()
                loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore() # 恢复embedding参数
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        if ema:
            ema.update()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probabilities, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1), running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy
