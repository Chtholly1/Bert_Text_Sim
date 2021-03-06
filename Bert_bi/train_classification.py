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
from transformers import BertTokenizer, get_linear_schedule_with_warmup as linear_warmup_schedule
from transformers.optimization import AdamW

#from business.model_plant import train, validate, train_mixup
from business.model_plant_classification import train, validate
from business.dataprocess.data_utils import MyDataSet, load_data
from business.models.model import AlBertModel, AlBertEncode, AlBertClassification
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
    print(total_data[:10])
    train_data = total_data[:train_data_max_num]
    train_dataset = MyDataSet(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size)
    print(len(train_loader))

    logging.info("\t* Loading validation data...")
    val_data = load_data(args.val_file)
    val_data = val_data[:val_data_max_num]
    val_dataset = MyDataSet(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)

    logging.info("\t* Building model...")
    config = Config()
    if MODEl_NAME.find('base_albert') >= 0:
        model = AlBertClassification(config).to(device)
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
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=0)
    scheduler1 = linear_warmup_schedule(optimizer, num_warmup_steps=0.1*len(train_loader)*args.epochs, num_training_steps = len(train_loader)*args.epochs)

    best_score = 8888
    start_epoch = 1

    #????????????EMA, FMG?????????????????????????????????
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
    # ?????????????????????????????????????????????
    _, valid_loss, valid_accuracy = validate(model, val_loader, device, ema=ema)
    logging.info("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%".format(valid_loss, (valid_accuracy * 100)))
    logging.info("\n" + 20 * "=" + "Training Bert model on device: {}".format(device) + 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, args.epochs + 1):
        logging.info("* Training epoch {}:".format(epoch))
        print(optimizer.param_groups[0]['lr'])
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer,
                                                       args.max_grad_norm, device, scheduler1, fgm=fgm, pgd=pgd, ema=ema)

        logging.info("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
                     .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        logging.info("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = validate(model, val_loader, device, ema=ema)
        logging.info("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
                     .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))

        #scheduler.step(epoch_accuracy)
        #scheduler.step(epoch_loss)
        if epoch_loss > best_score:
            patience_counter += 1
        else:
            best_score = epoch_loss
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


