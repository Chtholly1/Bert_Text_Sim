import os
import time
import torch
from transformers import BertTokenizerFast, AlbertModel, AlbertConfig

now_time = time.strftime("%Y%m%d%H", time.localtime())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
root_path = "../"
resource_path = root_path + 'resource/'
model_path = resource_path + 'base_models/base_albert/'
data_path = resource_path + 'data/'

save_path = "./model_saved/best_model.pth.%s"%(now_time)
load_path = './model_saved/best_model.pth.2022041310'

tokenizer = BertTokenizerFast.from_pretrained(model_path)
Config = AlbertConfig.from_pretrained(model_path)

batch_size = 128
maxlen = 128
epochs = 5

#Config.attention_probs_dropout_prob = 0.3
train_data_max_num = 100000
val_data_max_num = 5000

#output_way = 'pooler'
output_way = 'cls'
assert output_way in ['pooler','cls']

sts_file_path = data_path+ "STS-B/"
sts_train_file = 'cnsd-sts-train.txt'
sts_test_file = 'cnsd-sts-test.txt'
sts_dev_file = 'cnsd-sts-dev.txt'

snli_file_path = data_path + "SNLI/"
snli_train_file = 'cnsd_snli_v1.0.train.jsonl'
snli_val_file = 'cnsd_snli_v1.0.dev.jsonl'
