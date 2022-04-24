# -*-coding:utf-8-*-
import time
import math
import torch
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AlbertForSequenceClassification, BertForSequenceClassification, AlbertModel, BertForMaskedLM, AlbertPreTrainedModel

from conf.config import categories, MODEl_NAME

class AlBertModelSCL(nn.Module):
    def __init__(self, config, device):
        super(AlBertModelSCL, self).__init__()
        self.device = device
        self.temperature=0.05
        self.lamb = 0.2
        self.num_labels = len(categories)
        #self.bert = AlbertForSequenceClassification.from_pretrained(config.model_name, num_labels=len(categories))  # /bert_pretrain
        self.bert = AlbertModel.from_pretrained(config.model_name, num_labels=len(categories)) 
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, len(categories))
        for param in self.bert.parameters():
            param.requires_grad = True 

    def forward(self, input_ids, att_mask, labels):
        last_hidden, pooled_out = self.bert(input_ids=input_ids, attention_mask=att_mask)[:2]
        #Compute CE loss
        pooled_output = self.dropout(pooled_out)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #Compute SCL loss
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        seq_embed = pooled_out
        seq_embed_l2 = torch.norm(seq_embed,dim = 1, keepdim=True)
        seq_embed = seq_embed/seq_embed_l2
        seq_embed_dot = torch.div(torch.matmul(seq_embed, seq_embed.T), self.temperature)
        #logits_max, _ = torch.max(seq_embed_dot, dim=1, keepdim=True)
        #scl_logits = seq_embed_dot - logits_max.detach()
        scl_logits = seq_embed_dot + torch.tensor(1e-6)
        #print(scl_logits)
        logits_mask = torch.scatter(
            torch.ones_like(mask), #被修改的张量
            1,#沿着dim=1这个维度进行修改
            torch.arange(mask.shape[0]).view(-1, 1).to(self.device), #被修改元素的索引
            0 #重新赋值
        )
        exp_logits  = torch.exp(scl_logits) * logits_mask
        #log_prob = scl_logits - torch.log(exp_logits.sum(1, keepdim=True)+ torch.tensor(1e-6))
        log_prob = scl_logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask = mask * logits_mask
        mean_log_prob_pos = -(mask * log_prob).sum(1) / (mask.sum(1)+torch.tensor(1e-6))
        scl_loss = mean_log_prob_pos.mean()
        final_loss = (1-self.lamb)*loss+ self.lamb*scl_loss
        #print(mean_log_prob_pos)

        probabilities= t.softmax(logits, dim=-1)
        return final_loss, logits, probabilities

class AlBertModelMixup(AlbertPreTrainedModel):

    def __init__(self, config):
        super(AlBertModelMixup, self).__init__(config)
        #self.bert = AlbertForSequenceClassification.from_pretrained(config.model_name, num_labels=len(categories))  # /bert_pretrain
        self.num_labels = len(categories)
        self.bert = AlbertModel.from_pretrained(MODEl_NAME, num_labels=self.num_labels)  # /bert_pretrain
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, len(categories))
        self.loss_fct = CrossEntropyLoss()
        for param in self.bert.parameters():
            param.requires_grad = True 
        #self.init_weights()
    
    def validate(self, input_ids, att_mask, label_ids): 
        last_hidden_state, pooler_out = self.bert(input_ids=input_ids, attention_mask=att_mask)[:2]
        pooler_out = self.dropout(pooler_out)
        logits = self.classifier(pooler_out)
        probabilities = t.softmax(logits, dim=-1)
        loss = self.loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))

        return loss, logits, probabilities       
    
    def forward(self, input_ids, att_mask, label_ids):
        last_hidden_state, pooler_out = self.bert(input_ids=input_ids, attention_mask=att_mask)[:2]
        one_hot_label = t.nn.functional.one_hot(label_ids, num_classes=self.num_labels)
        pooler_out = self.dropout(pooler_out)
        reverse_idx = list(range(input_ids.shape[0]))[::-1]
        lamd = np.random.beta(1,1)
        #lamd=0.5
        new_pooled_out = lamd*pooler_out + (1-lamd)*pooler_out[reverse_idx]
        new_label = lamd*one_hot_label + (1-lamd)*one_hot_label[reverse_idx]
        new_pooled_out = t.cat((pooler_out, new_pooled_out[:int(input_ids.shape[0]/2)]), dim=0)
        new_label = t.cat((one_hot_label, new_label[:int(input_ids.shape[0]/2)]), dim=0)

        logits = self.classifier(new_pooled_out)
        probabilities = t.softmax(logits, dim=-1)
        loss = new_label*t.log(probabilities+1e-6)
        loss = t.sum(loss, dim=1)
        loss = -loss.mean()
        return loss, logits[:input_ids.shape[0]], probabilities[:input_ids.shape[0]]

class AlBertModel(nn.Module):
    def __init__(self, config):
        super(AlBertModel, self).__init__()
        self.bert = AlbertForSequenceClassification.from_pretrained(config.model_name, num_labels=len(categories))  # /bert_pretrain
        for param in self.bert.parameters():
            param.requires_grad = True 
        
    def encode(self, input_ids, att_mask, token_type_ids, label_ids):
        loss, logits, hidden_states = self.bert(input_ids=input_ids, attention_mask=att_mask,
                                                labels=label_ids, token_type_ids=token_type_ids, 
                                                output_hidden_states=True)[:3]
        return hidden_states[-1]

    def forward(self, input_ids, att_mask, token_type_ids, label_ids):
        loss, logits, hidden_states = self.bert(input_ids=input_ids, attention_mask=att_mask,
                                                labels=label_ids, token_type_ids=token_type_ids, 
                                                output_hidden_states=True)[:3]
        probabilities = t.softmax(logits, dim=-1)
        return loss, logits, probabilities

class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(config.model_name, num_labels=len(categories))  # /bert_pretrain
        for param in self.bert.parameters():
            param.requires_grad = True 
        
    def forward(self, input_ids, att_mask, label_ids):
        loss, logits, hidden_states = self.bert(input_ids=input_ids, attention_mask=att_mask,
                                                labels=label_ids,
                                                output_hidden_states=True)[:3]
        probabilities = t.softmax(logits, dim=-1)
        return loss, logits, probabilities

class BertMLM(nn.Module):
    def __init__(self, config):
        super(BertMLM, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(config.model_name, num_labels=2)  # /bert_pretrain
        for param in self.bert.parameters():
            param.requires_grad = True  

    def forward(self, input_ids, att_mask, label_ids):
        outputs = self.bert(input_ids = input_ids, attention_mask=att_mask, labels=label_ids)
        loss = outputs.loss
        #logits = outputs.logits
        return loss

class AlBertModelCNNMult(nn.Module):
    def __init__(self, config):
        super(AlBertModelCNNMult, self).__init__()
        self.bert = AlbertModel.from_pretrained(config.model_name, num_labels=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.convs1 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embedding_size,
                                    out_channels=config.out_channels,
                                    kernel_size=h),
                          nn.BatchNorm1d(config.out_channels),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_text_len - h + 1), )
            for h in config.kernel_size
        ])
        self.cnn_merge = nn.Sequential(nn.Conv1d(in_channels=config.out_channels * len(config.kernel_size), 
                                                out_channels = config.out_channels*len(config.kernel_size),
                                                kernel_size =2),
                                      nn.BatchNorm1d(config.out_channels *len(config.kernel_size)),
                                      nn.ReLU(),
                                      nn.MaxPool1d(4-2+1), )

        self.merge_features = nn.Linear(config.out_channels * len(config.kernel_size) + config.extra_feature_num,
                                        config.out_channels)
        self.classify = nn.Linear(config.out_channels, 2)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, att_mask, token_type_ids, labels, course_tensor):
        out_list = []
        new_input_ids = input_ids.permute(1,0,2)
        new_att_mask = att_mask.permute(1,0,2)
        new_token_type_ids = token_type_ids.permute(1,0,2)
        for i in range(input_ids.shape[1]):
            input_ids_i = new_input_ids[i].view(-1, input_ids.shape[2])
            att_mask_i = new_att_mask[i].view(-1, att_mask.shape[2])
            token_type_ids_i = new_token_type_ids[i].view(-1, token_type_ids.shape[2])
            seq_out, pooled_out = self.bert(input_ids=input_ids_i, attention_mask=att_mask_i, token_type_ids=token_type_ids_i)[:2]
            out_list.append(pooled_out)
        output = t.stack(out_list, dim=1)
        output = self.cnn_merge(output.permute(0,2,1))
        output = output.view(-1, output.size(1))
        output = t.cat((output, course_tensor), 1)
        output = self.merge_features(output)
        output = self.classify(output)
        probabilities = t.softmax(output, dim=-1)
        loss = self.loss_fct(output.view(-1, 2), labels.view(-1))
        return loss, output, probabilities

class AlBertModelCNN(nn.Module):
    def __init__(self, config):
        super(AlBertModelCNN, self).__init__()
        self.bert = AlbertForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.convs1 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embedding_size,
                                    out_channels=config.out_channels,
                                    kernel_size=h),
                          nn.BatchNorm1d(config.out_channels),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_text_len - h + 1), )
            for h in config.kernel_size
        ])
        self.merge_features = nn.Linear(config.out_channels * len(config.kernel_size) + config.extra_feature_num,
                                        config.out_channels)
        self.classify = nn.Linear(config.out_channels, 2)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, att_mask, token_type_ids, labels, course_tensor):
        loss, logits, hidden_states = self.bert(input_ids=input_ids, attention_mask=att_mask,
                                                token_type_ids=token_type_ids, labels=labels,
                                                output_hidden_states=True)[:3]

        embed_x1 = hidden_states[-1].permute(0, 2, 1)
        out1 = [conv(embed_x1) for conv in self.convs1]
        out1 = t.cat(out1, dim=1)
        out1 = out1.view(-1, out1.size(1))
        output = t.cat((out1, course_tensor), 1)
        output = self.merge_features(output)
        output = self.classify(output)
        probabilities = t.softmax(output, dim=-1)
        loss = self.loss_fct(output.view(-1, 2), labels.view(-1))
        return loss, output, probabilities


class BertModelCNN(nn.Module):
    def __init__(self, config):
        super(BertModelCNN, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(config.model_name, num_labels=2)  # /bert_pretrain/
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.convs1 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embedding_size,
                                    out_channels=config.out_channels,
                                    kernel_size=h),
                          nn.BatchNorm1d(config.out_channels),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_text_len - h + 1), )
            for h in config.kernel_size
        ])
        self.merge_features = nn.Linear(config.out_channels * len(config.kernel_size) + config.extra_feature_num,
                                        config.out_channels)
        self.classify = nn.Linear(config.out_channels, 2)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels, course_tensor):
        loss, logits, hidden_states = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                                                token_type_ids=batch_seq_segments, labels=labels,
                                                output_hidden_states=True)[:3]

        embed_x1 = hidden_states[-1].permute(0, 2, 1)
        out1 = [conv(embed_x1) for conv in self.convs1]
        out1 = t.cat(out1, dim=1)
        out1 = out1.view(-1, out1.size(1))
        output = t.cat((out1, course_tensor), 1)
        output = self.merge_features(output)
        output = self.classify(output)
        probabilities = t.softmax(output, dim=-1)
        loss = self.loss_fct(output.view(-1, 2), labels.view(-1))
        return loss, output, probabilities
