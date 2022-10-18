from utils import (
    df2train_test_dfs, basic_tokenizer, init_weights, count_parameters,
    translate_sentence, display_attention, df2train_valid_test_dfs,
    save_model, load_model, df2train_error_dfs, word2chars
)
from models import Encoder, Decoder, Attention, Seq2Seq
from pipeline import train, test_accuracy
from inference import test_beam, test_greedy
from focalLoss import FocalLoss

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
import random
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import time
# from torchtext.data.metrics import bleu_score

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

import numpy as np
import math
import time
import sys

import warnings as wrn
wrn.filterwarnings('ignore')


def error_df(df, error='Cognitive Error'):
    df = df.loc[df['ErrorType'] == error]
    df['Word'] = df['Word'].apply(word2chars)
    df['Error'] = df['Error'].apply(word2chars)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.iloc[:, [1, 0]]
    df.to_csv('./Dataset/error.csv', index=False)


def check_error():
    df = pd.read_csv('./Dataset/sec_dataset_II.csv')
    df = df.iloc[:, :]
    # df2train_test_dfs(df=df, test_size=0.15)
    df2train_valid_test_dfs(df=df, test_size=0.15)

    # ['Cognitive Error', 'Homonym Error', 'Run-on Error',
    #  'Split-word Error (Left)', 'Split-word Error (Random)',
    #  'Split-word Error (Right)', 'Split-word Error (both)',
    #  'Typo (Avro) Substituition', 'Typo (Bijoy) Substituition',
    #  'Typo Deletion', 'Typo Insertion', 'Typo Transposition',
    #  'Visual Error', 'Visual Error (Combined Character)']
    error_name = 'Cognitive Error'
    error_df(df, error_name)
    # df2train_error_dfs(df, error='Cognitive Error')
    # sys.exit()

    SRC = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>',
        sequential=True, use_vocab=True, include_lengths=True
    )
    TRG = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>',
        sequential=True, use_vocab=True
    )
    fields = {
        'Error': ('src', SRC),
        'Word': ('trg', TRG)
    }
    train_data, valid_data, test_data = TabularDataset.splits(
        path='./Dataset',
        train='train.csv',
        validation='valid.csv',
        test='test.csv',
        format='csv',
        fields=fields
    )
    error_data, _ = TabularDataset.splits(
        path='./Dataset',
        train='error.csv',
        test='error.csv',
        format='csv',
        fields=fields
    )

    # print(error_data)
    # sys.exit()

    SRC.build_vocab(train_data, max_size=64, min_freq=100)
    TRG.build_vocab(train_data, max_size=64, min_freq=75)
    # print(len(SRC.vocab), len(TRG.vocab))

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 256
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 64
    DEC_EMB_DIM = 64
    ENC_HIDDEN_DIM = 256
    DEC_HIDDEN_DIM = 512
    ENC_DROPOUT = 0.25
    DEC_DROPOUT = 0.25
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    MAX_LEN = 32
    N_EPOCHS = 10
    CLIP = 1
    PATH = ''

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=DEVICE
    )
    error_iterator, _ = BucketIterator.splits(
        (error_data, error_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=DEVICE
    )

    attention = Attention(ENC_HIDDEN_DIM, DEC_HIDDEN_DIM)
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HIDDEN_DIM, DEC_HIDDEN_DIM, ENC_DROPOUT)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HIDDEN_DIM, DEC_HIDDEN_DIM, DEC_DROPOUT, attention)

    model = Seq2Seq(encoder, decoder, SRC_PAD_IDX, DEVICE).to(DEVICE)
    model.apply(init_weights)
    # print(model)
    # print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    # scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=4)

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    # criterion = nn.NLLLoss(ignore_index=TRG_PAD_IDX)
    # criterion = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')

    PATH = './Checkpoints/spell_s2s.pth'
    # best_loss = 1e10

    checkpoint, epoch, train_loss = load_model(model, optimizer, PATH)
    # best_loss = train_loss
    error_df_ = pd.read_csv('./Dataset/error.csv')
    error_pct = (len(error_df_) / len(df)) * 100
    
    print(f"\n------------\nError Name: {error_name} - {error_pct:.2f}% of dataset\n------------")
    test_accuracy(error_data, SRC, TRG, model, DEVICE)


    # test_beam(model, train_data, test_data, SRC, TRG, DEVICE)
    # test_greedy(test_data, SRC, TRG, model, DEVICE)

    # example_idx = 1
    # src = vars(train_data.examples[example_idx])['src']
    # trg = vars(train_data.examples[example_idx])['trg']
    # print(f'src = {src}')
    # print(f'trg = {trg}')
    # translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)
    # print(f'predicted trg = {translation}')
    # display_attention(src, translation, attention)


if __name__ == '__main__':
    check_error()
