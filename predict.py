import pickle
import time
import codecs
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


def test(model_path, data_path, split, batch_size, max_sent_length, cuda):
    device = torch.device("cuda:0" if cuda else "cpu")
    dataset = SIGHAN(split='dev', root_path=data_path)   
    model = CharWordSeg.load(model_path)
    model = model.to(device)
    model.eval()

    vocab_tag = model.vocab_tag
    tok2idx, idx2tok = vocab_tag['token_to_index'], vocab_tag['index_to_token']
    tag2idx, idx2tag = vocab_tag['tag_to_index'], vocab_tag['index_to_tag']

    data_loader = DataLoader(dataset, batch_size=batch_size)
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

            loss = -model(input_chars, target_tags, tags_mask)
            val_loss += loss
            cum_cnt = cum_cnt + input_chars.shape[0] 
    val_loss = val_loss / cum_cnt
    return val_loss

def create_output(chars_batch, tags_batch):
    output = []
    for seq, tag in zip(chars_batch, tags_batch):
        new_sent = []
        for char, tag in zip(seq, tag):
            new_sent.append(char)
            if tag=='S' or tag=='E':
                new_sent.append("  ")
        output.append("".join(new_sent).strip())
    return output

def predict(model_path, data_path, split, output_path, batch_size, max_sent_length, cuda):
    device = torch.device("cuda:0" if cuda else "cpu")
    dataset = SIGHAN(split=split, root_path=data_path)   
    output = []

    model = CharWordSeg.load(model_path)
    model = model.to(device)
    model.eval()

    vocab_tag = model.vocab_tag
    tok2idx, idx2tok = vocab_tag['token_to_index'], vocab_tag['index_to_token']
    tag2idx, idx2tag = vocab_tag['tag_to_index'], vocab_tag['index_to_tag']

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():   
        for iter_idx, data in enumerate(data_loader):
            chars = get_feature(data, stage='predict')
            chars_batch, chars_mask = pad_sents(chars, pad_token, max_len=max_sent_length)

            input_chars = torch.tensor(token2id(chars_batch, tok2idx, unk_id=tok2idx['<UNK>']), device=device)
            chars_mask = torch.tensor(chars_mask, dtype=torch.uint8, device=device)

            pred_tags = id2token(model.decode(input_chars, chars_mask), idx2tag)
            output.extend(create_output(chars, pred_tags))

    with codecs.open(output_path, 'w', 'utf8') as f:
        for sent in output:
            print(sent, file=f)


@click.command()
@click.option('--mode', default='pred', help="eval - compute the test loss; pred - generate the output textfiles")
@click.option('--model_path', default="outputs/model_best", help="path of the trained model", required=True)
@click.option('--data_path', default="data/datasets/sighan2005-pku", help="path of the training data", required=True)
@click.option('--split', default="test")
@click.option('--output_path', default="test_output.txt")
@click.option('--batch_size', default=1024, type=int)
@click.option('--max_sent_length', default=200, type=int)
@click.option('--cuda', default=True, help="whether to use cuda", type=bool, required=True)
def main(mode, model_path, data_path, split, output_path, batch_size, max_sent_length, cuda):
    if mode == 'eval':
        print(f"model:{model_path}\t data:{data_path} split:{split}")
        print(f"loss: {test(model_path, data_path, split, batch_size, max_sent_length, cuda)}")
    elif mode == 'pred':
        predict(model_path, data_path, split, output_path, batch_size, max_sent_length, cuda)

if __name__ == '__main__':
    main() # pyline:  disable=no-value-for-argument
