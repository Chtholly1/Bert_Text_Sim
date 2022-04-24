# -*- coding: utf-8 -*-
import os
import time
import random
import argparse
import logging
import warnings
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers.optimization import AdamW

from business.model_plant import train, validate, train_mixup
from business.dataprocess.data_utils import MyDataSet, load_data
from business.models.model import AlBertModelSCL, AlBertModel, BertModel, AlBertModelMixup
from business.tools import EMA, FGM, PGD, setup_seed
from conf.config import Config, args, MODEl_NAME, train_data_max_num, val_data_max_num

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
warnings.simplefilter(action='ignore')
now_time = time.strftime("%Y%m%d%H", time.localtime())
torch.set_printoptions(profile="full")

def main(args):
    setup_seed(2000)
    device = torch.device("cuda:{}".format(args.gpu_index) if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(MODEl_NAME, do_lower_case=True)

    logging.info(20 * "=" + " Preparing for training " + 20 * "=")
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    logging.info("\t* Loading training data...")
    total_data = load_data(args.train_file)
    random.shuffle(total_data)
    train_data = total_data[:train_data_max_num]
    train_dataset = MyDataSet(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    logging.info("\t* Loading validation data...")
    val_data = load_data(args.val_file)
    val_data = val_data[:val_data_max_num]
    val_dataset = MyDataSet(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)

    logging.info("\t* Building model...")
    config = Config()
    if MODEl_NAME.find('base_albert') >= 0:
        model = AlBertModel(config).to(device)
    elif MODEl_NAME.find('base_roberta') >= 0:
        model = BertModel(config).to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=0)

    best_score = 0.0
    start_epoch = 1

    #是否使用EMA, FMG等提升最终指标的技巧。
    if args.use_EMA:
        ema = EMA(model, 0.999)
        ema.register()
    else:
        ema = None

    fgm = None
    pgd = None
    if args.attack_type=='FGM':
        fgm = FGM(model, epsilon=1, emb_name='word_embeddings')
    elif args.attack_type == 'PGD':
        pgd = PGD(model, emb_name='word_embeddings')
    #是否在模型初始化时先测试其效果
    _, valid_loss, valid_accuracy = validate(model, val_loader, device, ema=ema)
    logging.info("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%".format(valid_loss, (valid_accuracy * 100)))
    logging.info("\n" + 20 * "=" + "Training Bert model on device: {}".format(device) + 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, args.epochs + 1):
        logging.info("* Training epoch {}:".format(epoch))

        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer,
                                                       args.max_grad_norm, device, fgm=fgm, pgd=pgd, ema=ema)
        #epoch_time, epoch_loss, epoch_accuracy = train_mixup(model, train_loader, optimizer,
        #                                               args.max_grad_norm, device, fgm=fgm, pgd=pgd, ema=ema)

        logging.info("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
                     .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        logging.info("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = validate(model, val_loader, device, ema=ema)
        logging.info("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
                     .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))

        scheduler.step(epoch_accuracy)

        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            if ema:
                ema.apply_shadow()
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score},
                       os.path.join(args.target_dir, "best.pth.tar.%s" % now_time))
            if ema:
                ema.restore()
        if patience_counter >= args.patience:
            logging.info("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == "__main__":
    for k in args.__dict__:
        logging.info(k + ": " + str(args.__dict__[k]))

    main(args)


