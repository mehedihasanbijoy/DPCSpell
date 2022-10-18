from utils import (
    df2train_test_dfs, basic_tokenizer, init_weights, count_parameters,
    translate_sentence, display_attention, df2train_valid_test_dfs,
    save_model, load_model, df2train_error_dfs
)
from models import Encoder, Decoder, Attention, Seq2Seq
from pipeline import train, test_accuracy
from inference import test_beam, test_greedy
from focalLoss import FocalLoss
from errors import error_df

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
import random
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

import numpy as np
import math
import time
import sys
import os
import argparse

import warnings as wrn
wrn.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CORPUS", help="Path of the Corpus", type=str, default="./Dataset/corpus2.csv", 
        choices=["./Dataset/corpus.csv", "./Dataset/corpus2.csv"]
    )
    parser.add_argument("--ENC_EMB_DIM", help="Encoder Embedding Dimension", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--DEC_EMB_DIM", help="Decoder Embedding Dimension", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--ENC_HIDDEN_DIM", help="Encoder Hidden Dimension", type=int,default=256, choices=[128, 256, 512])
    parser.add_argument("--DEC_HIDDEN_DIM", help="Decoder Hidden Dimension", type=int, default=512, choices=[256, 512, 1024])
    parser.add_argument("--ENC_DROPOUT", help="Encoder Dropout Ratio", type=float, default=0.1, choices=[0.1, 0.2, 0.5])
    parser.add_argument("--DEC_DROPOUT", help="Decoder Dropout Ratio", type=float, default=0.1, choices=[0.1, 0.2, 0.5])
    parser.add_argument("--MAX_LEN", help="Maximum Length", type=int, default=48, choices=[48, 56, 64])
    parser.add_argument("--BATCH_SIZE", help="Batch Size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--CLIP", help="Gradient Clipping", type=float, default=1, choices=[0.1, 0.2, 0.5, 1])
    parser.add_argument("--N_EPOCHS", help="Number of Epochs", type=int, default=100)
    parser.add_argument("--LEARNING_RATE", help="Learning Rate", type=float, default=0.0005, choices=[0.0005, 0.00005, 0.000005])
    args = parser.parse_args()


    df = pd.read_csv(args.CORPUS)
    df2train_valid_test_dfs(df=df, test_size=0.15)

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

    SRC.build_vocab(train_data, max_size=64, min_freq=100)
    TRG.build_vocab(train_data, max_size=64, min_freq=75)
    # -------------------------------------
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = args.BATCH_SIZE
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = args.ENC_EMB_DIM
    DEC_EMB_DIM = args.DEC_EMB_DIM
    ENC_HIDDEN_DIM = args.ENC_HIDDEN_DIM
    DEC_HIDDEN_DIM = args.DEC_HIDDEN_DIM
    ENC_DROPOUT = args.ENC_DROPOUT
    DEC_DROPOUT = args.DEC_DROPOUT
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    MAX_LEN = args.MAX_LEN
    N_EPOCHS = args.N_EPOCHS
    CLIP = args.CLIP
    # -------------------------------------
    PATH = './Checkpoints/GRUSeq2Seq.pth'

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
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
    # print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    # scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=4)

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    # criterion = nn.NLLLoss(ignore_index=TRG_PAD_IDX)
    # criterion = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')

    best_loss = 1e10
    epoch = 1
    if os.path.exists(PATH):
        checkpoint, epoch, train_loss = load_model(model, optimizer, PATH)
        best_loss = train_loss

    for epoch in range(epoch, N_EPOCHS):
        print(f'Epoch: {epoch} / {N_EPOCHS}')
        train_loss = train(model, train_iterator, optimizer, criterion)
        print(f"Train Loss: {train_loss:.2f}")

        if train_loss < best_loss:
            best_loss = train_loss
            save_model(model, epoch, optimizer, train_loss, PATH)

        # scheduler.step()
        # if epoch%10 == 0:
        #     # test_accuracy(valid_data, SRC, TRG, model, DEVICE)
        #     test_accuracy(error_data, SRC, TRG, model, DEVICE)

    test_accuracy(valid_data, SRC, TRG, model, DEVICE)

    
    # errors = ['Cognitive Error', 'Homonym Error', 'Run-on Error',
    #  'Split-word Error (Left)', 'Split-word Error (Random)',
    #  'Split-word Error (Right)', 'Split-word Error (both)',
    #  'Typo (Avro) Substituition', 'Typo (Bijoy) Substituition',
    #  'Typo Deletion', 'Typo Insertion', 'Typo Transposition',
    #  'Visual Error', 'Visual Error (Combined Character)']

    # for error in errors:
    #     print(f"-----\nError Type: {error}\n-----")
    #     error_df(df, error)
    #     error_data, _ = TabularDataset.splits(
    #         path='./Dataset',
    #         train='error.csv',
    #         test='error.csv',
    #         format='csv',
    #         fields=fields
    #     )
    #     eval_df = test_accuracy(error_data, SRC, TRG, model, DEVICE)
    #     error = error.replace(' ', '').replace('(', '').replace(')', '')
    #     eval_df.to_csv(f'./Corrections/s2sJL_{error}.csv')
    #     print('\n\n')


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
    main()
