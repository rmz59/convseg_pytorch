import pickle
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import click

from itertools import chain
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader
from crf import CRF
from utils import SIGHAN, build_vocab, add_id, get_feature, token2id, id2token, pad_sents
from model import CharWordSeg

unk_token = "<UNK>"
pad_token = "<PAD>"
logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s', level=logging.DEBUG)

def evaluate(model, dev_data, batch_size, max_sent_length, device):
    was_training = model.training
    model.eval()

    vocab_tag = model.vocab_tag
    tok2idx, idx2tok = vocab_tag['token_to_index'], vocab_tag['index_to_token']
    tag2idx, idx2tag = vocab_tag['tag_to_index'], vocab_tag['index_to_tag']

    data_loader = DataLoader(dev_data, batch_size=batch_size)
    val_loss = 0
    cum_cnt = 0
    with torch.no_grad():   
        for iter_idx, data in enumerate(data_loader):
            chars_batch, tags_batch = get_feature(data)
            chars_batch, chars_mask = pad_sents(chars_batch, pad_token, max_len=max_sent_length)
            tags_batch, tags_mask = pad_sents(tags_batch, pad_token, max_len=max_sent_length)

            input_chars = torch.tensor(token2id(chars_batch, tok2idx, unk_id=tok2idx['<UNK>']), device=device)
            target_tags = torch.tensor(token2id(tags_batch, tag2idx), device=device)
            tags_mask = torch.tensor(tags_mask, dtype=torch.uint8, device=device)

#            output_emission = conv_seg(input_chars.to(device))
#            loss = -crf_model(output_emission.transpose(0,1), target_tags.transpose(0,1).to(device), tags_mask.transpose(0,1).to(device))
            loss = -model(input_chars, target_tags, tags_mask)
            val_loss += loss
            cum_cnt = cum_cnt + input_chars.shape[0] 
    val_loss = val_loss / cum_cnt

    if was_training:
        model.train()

    return val_loss

@click.command()
@click.option('--split', default='train', required=True)
@click.option('--data_path', default="data/datasets/sighan2005-pku", help="path of the training data", required=True)
@click.option('--save_to', default="outputs", required=True)
@click.option('--batch_size', default=2048, type=int)
@click.option('--max_sent_length', default=200, type=int)
@click.option('--char_embed_size', default=200, type=int)
@click.option('--num_hidden_layer', default=5, type=int)
@click.option('--channel_size', default=200, type=int)
@click.option('--kernel_size', default=3, type=int)
@click.option('--learning_rate', default=2e-3, type=float)
@click.option('--max_epoch', default=1000, type=int)
@click.option('--log_every', default=100, type=int)
@click.option('--patience', default=20, type=int)
@click.option('--max_num_trial', default=20, type=int)
@click.option('--lr_decay', default=0.5, type=float)
@click.option('--last_model_path', default=None, type=str)
@click.option('--cuda', default=True, type=bool)
def train(split, data_path, save_to, batch_size, max_sent_length, char_embed_size, num_hidden_layer, 
          channel_size, kernel_size, learning_rate, max_epoch, log_every, patience, max_num_trial, lr_decay, last_model_path, cuda): 
    if not Path(save_to).exists():
        Path(save_to).mkdir()

    device = torch.device("cuda:0" if cuda else "cpu")
    dataset = SIGHAN(split=split, root_path=data_path)   
    val_dataset = SIGHAN(split='dev', root_path=data_path)   

    # build vocab
    train_chars, train_tags = get_feature(dataset.data)
    vocab = build_vocab(chain.from_iterable(train_chars))
    tok2idx, idx2tok = add_id(vocab)
    tags = build_vocab(["B","M","E","S"], add_unk=False, add_pad=True)
    tag2idx, idx2tag = add_id(tags)

    # build model
    # conv_seg = CharWordSeg(len(tags), len(vocab), char_embed_size, num_hidden_layer, channel_size, kernel_size).to(device)
    # crf_model = CRF(num_tags=5).to(device)
    # conv_seg.train()
    # crf_model.train()
    # optimizer = torch.optim.Adam(list(conv_seg.parameters())+list(crf_model.parameters()), lr=learning_rate)
    vocab_tag = {
        "token_to_index": tok2idx,
        "index_to_token": idx2tok,
        "tag_to_index": tag2idx,
        "index_to_tag": idx2tag
    }

    data_loader = DataLoader(dataset, batch_size=batch_size)
    model = CharWordSeg(vocab_tag, char_embed_size, num_hidden_layer, channel_size, kernel_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if last_model_path is not None:
        # load model
        logging.info(f'load model from  {last_model_path}')
        params = torch.load(last_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(params['state_dict'])
        logging.info('restore parameters of the optimizers')
        optimizer.load_state_dict(torch.load(last_model_path + '.optim'))

    model.train()

    epoch = 0
    cur_trial = 0 
    hist_valid_scores = []
    train_time = begin_time = time.time()
    logging.info("begin training!")
    while True:
        epoch += 1
        train_loss = 0
        cum_cnt = 0
        for iter_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            chars_batch, tags_batch = get_feature(data)
            chars_batch, chars_mask = pad_sents(chars_batch, pad_token, max_len=max_sent_length)
            tags_batch, tags_mask = pad_sents(tags_batch, pad_token, max_len=max_sent_length)

            input_chars = torch.tensor(token2id(chars_batch, tok2idx, unk_id=tok2idx['<UNK>']), device=device)
            target_tags = torch.tensor(token2id(tags_batch, tag2idx), device=device)
            tags_mask = torch.tensor(tags_mask, dtype=torch.uint8, device=device)

#            output_emission = conv_seg(input_chars.to(device))
#            loss = -crf_model(output_emission.transpose(0,1), target_tags.transpose(0,1).to(device), tags_mask.transpose(0,1).to(device))
            loss = -model(input_chars, target_tags, tags_mask)
            train_loss += loss
            cum_cnt = cum_cnt + input_chars.shape[0] 
            loss.backward()
            optimizer.step()

        train_loss = train_loss / cum_cnt
        val_loss = evaluate(model, val_dataset, batch_size, max_sent_length, device)


        logging.info(f'epoch {epoch}\t train_loss: {train_loss}\t val_loss:{val_loss}\t  speed:{time.time()-train_time:.2f}s/epoch\t time elapsed {time.time()-begin_time:.2f}s')
        train_time = time.time()
        
        is_better = len(hist_valid_scores) == 0 or val_loss < min(hist_valid_scores)
        hist_valid_scores.append(val_loss)
        
        if epoch % log_every == 0:
            model.save(f"{save_to}/model_step_{epoch}")
            torch.save(optimizer.state_dict(), f"{save_to}/model_step_{epoch}.optim")
        if is_better:
            cur_patience = 0
            model_save_path = f"{save_to}/model_best"
            print(f'save currently the best model to [{model_save_path}]')
            model.save(model_save_path)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), model_save_path + '.optim')
        elif cur_patience < patience:
            cur_patience += 1
            print('hit patience %d' % cur_patience)

            if cur_patience == patience:
                cur_trial += 1
                print(f'hit #{cur_trial} trial')
                if cur_trial == max_num_trial:
                    print('early stop!')
                    exit(0)

                # decay lr, and restore from previously best checkpoint
                lr = optimizer.param_groups[0]['lr'] * lr_decay
                logging.info(f'load previously best model and decay learning rate to {lr}')

                # load model
                params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                model.load_state_dict(params['state_dict'])
                model = model.to(device)

                logging.info('restore parameters of the optimizers')
                optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                # set new lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # reset patience
                cur_patience = 0

        if epoch == max_epoch:
            print('reached maximum number of epochs!')
            exit(0)


if __name__ == '__main__':
    train() # pyline:  disable=no-value-for-parameter