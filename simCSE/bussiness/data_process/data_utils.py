
import json
from torch.utils.data import Dataset

def load_snli_vocab(path):
    data = []
    with open(path) as f:
        for i in f:
            data.append([json.loads(i)['sentence1'], json.loads(i)['sentence2'], json.loads(i)['gold_label']])
    return data

def load_STS_data(path):
    data = []
    with open(path) as f:
        for i in f:
            d = i.strip().split("||")
            sentence1 = d[1]
            sentence2 = d[2]
            score = d[3]
            data.append([sentence1,sentence2,score])
    return data

def generate_treep_data(data):
    d = dict()
    res_list = []
    for item in data:
        sent1 = item[0]
        sent2 = item[1]
        label = item[2]
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

class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, maxlen, transform=None, target_transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.transform = transform
        self.target_transform = target_transform

    def text_to_id(self, source):
        sample = self.tokenizer([source[0],source[1],source[2]],max_length=self.maxlen,truncation=True,padding='max_length',return_tensors='pt')
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.text_to_id(self.data[idx])


class TestDataset:
    def __init__(self, data, tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.traget_idxs = self.text_to_id([x[0] for x in data])
        self.source_idxs = self.text_to_id([x[1] for x in data])
        self.label_list = [int(x[2]) for x in data]
        assert len(self.traget_idxs['input_ids']) == len(self.source_idxs['input_ids'])

    def text_to_id(self,source):
        sample = self.tokenizer(source,max_length=self.maxlen,truncation=True,padding='max_length',return_tensors='pt')
        return sample

    def get_data(self):
        return self.traget_idxs,self.source_idxs,self.label_list

