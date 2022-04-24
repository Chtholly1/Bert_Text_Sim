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
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from conf.config import args, device
from torch.nn import functional as F

loss_mse = torch.nn.MSELoss(reduction='mean')

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
 
def compute_mse_loss_test(pred0, pred1, pred2):
    sim_pos = F.cosine_similarity(pred0, pred1, dim=-1).cpu()
    sim_neg = F.cosine_similarity(pred0, pred2, dim=-1).cpu()
    label_pos = torch.ones(sim_pos.shape[0])
    label_neg = (-1)*torch.ones(sim_pos.shape[0])
    #prob_1 = 0.5*torch.cat((sim_pos, sim_neg), 0) +0.5
    prob_1 = torch.cat((sim_pos, sim_neg), 0)
    label = torch.cat((label_pos, label_neg), 0)
    loss = loss_mse(prob_1, label)
    correct_label = (prob_1.gt(0) == label).sum()

    prob = prob_1*0.5+0.5
    label_new = torch.cat((torch.ones(sim_pos.shape[0]), torch.zeros(sim_pos.shape[0])), 0)

    return loss, correct_label, prob, label

def compute_mse_loss(pred0, pred1, pred2):
    sim_pos = F.cosine_similarity(pred0, pred1, dim=-1)
    sim_neg = F.cosine_similarity(pred0, pred2, dim=-1)
    label_pos = torch.ones(sim_pos.shape[0])
    #label_neg = (-1)*torch.ones(sim_pos.shape[0])
    label_neg = torch.zeros(sim_pos.shape[0])
    prob_1 = torch.cat((sim_pos, sim_neg), 0) *0.5 + 0.5
    label = torch.cat((label_pos, label_neg), 0).to(device)
    loss = loss_mse(prob_1, label)
    correct_label = (prob_1.gt(0.5) == label).sum()
    return loss, correct_label
    
def compute_ce_loss(pred0, pred1, pred2):
    sim_pos = F.cosine_similarity(pred0, pred1, dim=-1).view(-1, 1).cpu()
    sim_neg = F.cosine_similarity(pred0, pred2, dim=-1).view(-1, 1).cpu()
    label_pos = torch.ones(sim_pos.shape[0])
    label_neg = torch.zeros(sim_pos.shape[0])
    prob_1 = 0.5*torch.cat((sim_pos, sim_neg), 0) +0.5
    prob_0 = 1 - prob_1
    prob = torch.cat((prob_0, prob_1), 1)
    label = torch.cat((label_pos, label_neg), 0).long()
    loss = F.cross_entropy(prob, label)
    correct_label = (prob_1.gt(0.5).view(-1) == label).sum()
    return loss, correct_label

def validate(model, dataloader, device, ema=None, output_file=None):
    # Switch to evaluate mode.
    model.eval()
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    all_labels = []
    all_prob = []
    tqdm_batch_iterator = tqdm(dataloader)
    with t.no_grad():
        for batch_index, data in enumerate(tqdm_batch_iterator):
            batch_start = time.time()
            input_ids0 = data['input_ids'][:,0].to(device)
            attention_mask0 = data['attention_mask'][:,0].to(device)
            token_type_ids0 = data['token_type_ids'][:,0].to(device)
            input_ids1 = data['input_ids'][:,1].to(device)
            attention_mask1 = data['attention_mask'][:,1].to(device)
            token_type_ids1 = data['token_type_ids'][:,1].to(device)
            input_ids2 = data['input_ids'][:,2].to(device)
            attention_mask2 = data['attention_mask'][:,2].to(device)
            token_type_ids2 = data['token_type_ids'][:,2].to(device)
            labels_pos = torch.ones(input_ids0.shape[0]).to(device)
            labels_neg = (-1)*torch.ones(input_ids0.shape[0]).to(device)
            prob_pos, loss_pos = model(input_ids0, attention_mask0, token_type_ids0, input_ids1, attention_mask1, token_type_ids1, labels_pos)
            prob_neg, loss_neg = model(input_ids0, attention_mask0, token_type_ids0, input_ids2, attention_mask2, token_type_ids2, labels_neg)
            all_labels.extend(torch.ones(input_ids0.shape[0]).tolist())
            all_labels.extend(torch.zeros(input_ids0.shape[0]).tolist())
            all_prob.extend(prob_pos.cpu().tolist())
            all_prob.extend(prob_neg.cpu().tolist())
            loss = loss_pos+loss_neg
            loss.mean()

            batch_time_avg += time.time() - batch_start
            running_loss += loss.item()
            description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
                .format(batch_time_avg / (batch_index + 1), running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_prob)
    return epoch_time, epoch_loss, auc

def test(model, dataloader, device, output_file=None):
    model.eval()
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    all_labels = []
    all_prob = []
    tqdm_batch_iterator = tqdm(dataloader)
    with t.no_grad():
        for batch_index, data in enumerate(tqdm_batch_iterator):
            batch_start = time.time()
            input_ids0 = data['input_ids'][:,0].to(device)
            attention_mask0 = data['attention_mask'][:,0].to(device)
            token_type_ids0 = data['token_type_ids'][:,0].to(device)
            input_ids1 = data['input_ids'][:,1].to(device)
            attention_mask1 = data['attention_mask'][:,1].to(device)
            token_type_ids1 = data['token_type_ids'][:,1].to(device)
            input_ids2 = data['input_ids'][:,2].to(device)
            attention_mask2 = data['attention_mask'][:,2].to(device)
            token_type_ids2 = data['token_type_ids'][:,2].to(device)
            pred0 = model(input_ids0, attention_mask0, token_type_ids0)
            pred1 = model(input_ids1, attention_mask1, token_type_ids1)
            pred2 = model(input_ids2, attention_mask2, token_type_ids2)
            loss, correct, prob, label = compute_mse_loss_test(pred0, pred1, pred2)
            all_labels.extend(label.tolist())
            all_prob.extend(prob.tolist())
            loss.mean()

            batch_time_avg += time.time() - batch_start
            running_loss += loss.item()
            correct_preds += correct.item()
            description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
                .format(batch_time_avg / (batch_index + 1), running_loss / (batch_index + 1))
            tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
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

def train(model, dataloader, optimizer, max_gradient_norm, device, schedule=None, fgm=None, pgd=None, ema=None):
    model.train()
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    all_labels = []
    all_prob = []
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, data in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        input_ids0 = data['input_ids'][:,0].to(device)
        attention_mask0 = data['attention_mask'][:,0].to(device)
        token_type_ids0 = data['token_type_ids'][:,0].to(device)
        input_ids1 = data['input_ids'][:,1].to(device)
        attention_mask1 = data['attention_mask'][:,1].to(device)
        token_type_ids1 = data['token_type_ids'][:,1].to(device)
        input_ids2 = data['input_ids'][:,2].to(device)
        attention_mask2 = data['attention_mask'][:,2].to(device)
        token_type_ids2 = data['token_type_ids'][:,2].to(device)
        optimizer.zero_grad()
        labels_pos = torch.ones(input_ids0.shape[0]).to(device)
        labels_neg = (-1)*torch.ones(input_ids0.shape[0]).to(device)
        #labels_neg = torch.zeros(input_ids0.shape[0]).to(device)
        prob_pos, loss_pos = model(input_ids0, attention_mask0, token_type_ids0, input_ids1, attention_mask1, token_type_ids1, labels_pos)
        prob_neg, loss_neg = model(input_ids0, attention_mask0, token_type_ids0, input_ids2, attention_mask2, token_type_ids2, labels_neg)
        all_labels.extend(torch.ones(input_ids0.shape[0]).tolist())
        all_labels.extend(torch.zeros(input_ids0.shape[0]).tolist())
        all_prob.extend(prob_pos.cpu().tolist())
        all_prob.extend(prob_neg.cpu().tolist())
        loss = loss_pos+loss_neg
        loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        if schedule is not None:
            schedule.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1), running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_prob)
    return epoch_time, epoch_loss, auc
