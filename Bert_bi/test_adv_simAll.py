# -*-coding:utf-8-*-
import os
import re
import random
import logging
import argparse
import warnings
import collections
from datetime import date,timedelta,datetime
import torch as t
import torch.nn as nn
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from business.model_plant import train, validate, test
from business.dataprocess.data_utils import MyDataSet, generate_all_stand, generate_all_data, load_data
from business.models.model import AlBertModel, AlBertEncode
from business.tools import EMA, FGM, setup_seed
from conf.config import Config, args, MODEl_NAME, train_data_max_num, val_data_max_num

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
warnings.filterwarnings(action='ignore')

def test1(args):
    setup_seed(2000)
    device = t.device("cuda:{}".format(args.gpu_index) if t.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name, do_lower_case=True)

    logging.info(20 * "=" + " Loading model from {} ".format(args.model) + 20 * "=")
    logging.info("\t* Loading validation data...")
    
    total_data = load_data(args.train_file)
    random.shuffle(total_data)
    train_data = total_data[:train_data_max_num]
    val_data = load_data(args.val_file)
    val_data = val_data[:val_data_max_num]
    val_dataset = MyDataSet(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)

    logging.info("\t* Building model...")
    config = Config()
    #model = AlBertModel(config).to(device)
    model = AlBertEncode(config).to(device)
    
    checkpoint = t.load(args.model, map_location=device)['model']
    print(t.load(args.model, map_location=device)['best_score'])
    #matrix = checkpoint.values()
    #name = [i[7:] for i in checkpoint.keys()]
    #state_dict_T = dict(zip(name,matrix))
    state_dict_T = checkpoint
    model.load_state_dict(state_dict_T, strict=True)
    model.eval()
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    date_time =  datetime.now().strftime("%Y%m%d%H")
    output_file = os.path.join(args.result_dir, 'test.' + date_time + '.csv')
    epoch_time, epoch_loss, epoch_accuracy, epoch_auc = test(model, val_loader, device, output_file=output_file)
    logging.info("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n"
                 .format(epoch_time, epoch_loss, (epoch_accuracy * 100), epoch_auc))


if __name__ == '__main__':
    test1(args)
