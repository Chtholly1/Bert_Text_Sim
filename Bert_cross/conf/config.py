#coding=utf-8

import os
import argparse

root_path = '../'
MODEl_NAME=os.path.join(root_path, 'resource/base_models/base_albert')
SNLI_data_path = os.path.join(root_path, 'resource/data/SNLI/')
model_saved_path = './resource/models/'
result_saved_path = './resource/result/'
#MODEl_NAME='./resource/base_models/base_roberta'

categories = ['yes', 'no']
train_data_max_num = 30000
val_data_max_num = 5000

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=MODEl_NAME, type=str)
parser.add_argument("--train_file", type=str, help="训练集文件", default=os.path.join(SNLI_data_path, 'cnsd_snli_v1.0.train.jsonl'))
parser.add_argument("--val_file", type=str, help="验证集文件", default=os.path.join(SNLI_data_path, 'cnsd_snli_v1.0.dev.jsonl'))
parser.add_argument("--target_dir", default=model_saved_path, type=str)
parser.add_argument("--result_dir", default=result_saved_path, type=str)
parser.add_argument("--model", default=os.path.join(model_saved_path, "best.pth.tar.2022040715"), type=str)
parser.add_argument("--max_length", default=128, type=int, help="截断的最大长度")
parser.add_argument("--epochs", default=25, type=int, help="最多训练多少个epoch")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--lr", default=2.5e-5, type=int)
parser.add_argument("--max_grad_norm", default=10.0, type=int)
parser.add_argument("--patience", default=3, type=int)
parser.add_argument("--gpu_index", default=1, type=int)
parser.add_argument("--attack_type", default='FGM', type=str)
parser.add_argument("--use_EMA", default=True, type=int)
parser.add_argument("--use_mixup", default=False, type=int)

args = parser.parse_args()

class Config:
    def __init__(self):
        self.num_labels = 2
        self.dropout_rate = 0.2
        self.num_attention_heads = 12
        self.attention_probs_dropout_prob = 0.1
        self.classifier_dropout_prob = 0.1
        self.hidden_size = 768
        self.final_out_size = 128
        self.vocab_size = 0
        self.embedding_size = 768
        self.out_channels = 256
        self.kernel_size = [2, 3, 4]
        self.max_text_len = 256
        self.cnn_conf_list = [ (3, 1), (3, 2), (3, 4), (3, 1)] 
        self.model_name = MODEl_NAME
