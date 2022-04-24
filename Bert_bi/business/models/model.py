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

class AlBertEncode(nn.Module):
    def __init__(self, config):
        super(AlBertEncode, self).__init__()
        self.bert = AlbertModel.from_pretrained(config.model_name)  # /bert_pretrain
        for param in self.bert.parameters():
            param.requires_grad = True 

    def forward(self, input_ids, att_mask, token_type_ids):
        last_hidden_state, pooler_out = self.bert(input_ids=input_ids, attention_mask=att_mask, token_type_ids=token_type_ids)[:2]
        return pooler_out

class AlBertCosine(nn.Module):
    def __init__(self, config):
        super(AlBertCosine, self).__init__()
        self.bert = AlbertModel.from_pretrained(config.model_name)  # /bert_pretrain
        self.loss_mse = torch.nn.MSELoss(reduction='mean')
        for param in self.bert.parameters():
            param.requires_grad = True 

    def forward(self, input_ids_0, att_mask_0, token_type_ids_0, input_ids_1, att_mask_1, token_type_ids_1, labels):
        last_hidden_state_0, pooler_out_0 = self.bert(input_ids=input_ids_0, attention_mask=att_mask_0, token_type_ids=token_type_ids_0)[:2]
        last_hidden_state_1, pooler_out_1 = self.bert(input_ids=input_ids_1, attention_mask=att_mask_1, token_type_ids=token_type_ids_1)[:2]
        sim = F.cosine_similarity(pooler_out_0, pooler_out_1, dim=-1)
        loss = self.loss_mse(sim, labels)
        #prob = 0.5*sim + 0.5
        return sim, loss

class AlBertClassification(nn.Module):
    def __init__(self, config):
        super(AlBertClassification, self).__init__()
        self.bert = AlbertModel.from_pretrained(config.model_name)  # /bert_pretrain
        self.cls = nn.Sequential(nn.Linear(3*config.hidden_size, config.hidden_size),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(config.hidden_size, 2),
                                )
        self.loss_fct = CrossEntropyLoss()

        for param in self.bert.parameters():
            param.requires_grad = True 

    def forward(self, input_ids_0, att_mask_0, token_type_ids_0, input_ids_1, att_mask_1, token_type_ids_1, labels):
        last_hidden_state_0, pooler_out_0 = self.bert(input_ids=input_ids_0, attention_mask=att_mask_0, token_type_ids=token_type_ids_0)[:2]
        last_hidden_state_1, pooler_out_1 = self.bert(input_ids=input_ids_1, attention_mask=att_mask_1, token_type_ids=token_type_ids_1)[:2]
        pooler_out = torch.cat((pooler_out_0, pooler_out_1, torch.abs(pooler_out_0-pooler_out_1)), dim=-1)
        out = self.cls(pooler_out)
        prob = t.softmax(out, dim=-1)
        loss = self.loss_fct(prob, labels)
        return prob, loss

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

