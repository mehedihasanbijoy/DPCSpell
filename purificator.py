from utils import (
    word2char, basic_tokenizer, count_parameters, initialize_weights,
    save_model, load_model, error_df, train_valid_test_df, mask2str,
    error_df_2, error_df_3, find_len, train_valid_test_df2, merge_dfs
)
from transformer import (
    Encoder, EncoderLayer, MultiHeadAttentionLayer,
    PositionwiseFeedforwardLayer, Decoder, DecoderLayer,
    Seq2Seq
)
from pipeline import train, evaluate
from metrics import evaluation_report, evaluation_report2, evaluation_report3

import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import torch
import torch.nn as nn
import os
import gc
import argparse
import sys

import warnings as wrn
wrn.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--HID_DIM", help="Hidden Dimension", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--ENC_LAYERS", help="Number of Encoder Layers", type=int, default=3, choices=[3, 5, 7])
    parser.add_argument("--DEC_LAYERS", help="Number of Decoder Layers", type=int,default=3, choices=[3, 5, 7])
    parser.add_argument("--ENC_HEADS", help="Number of Encoder Attention Heades", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--DEC_HEADS", help="Number of Decoder Attention Heades", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--ENC_PF_DIM", help="Encoder PF Dimension", type=int, default=256, choices=[64, 128, 256])
    parser.add_argument("--DEC_PF_DIM", help="Decoder PF Dimesnion", type=int, default=256, choices=[64, 128, 256])
    parser.add_argument("--ENC_DROPOUT", help="Encoder Dropout Ratio", type=float, default=0.1, choices=[0.1, 0.2, 0.5])
    parser.add_argument("--DEC_DROPOUT", help="Decoder Dropout Ratio", type=float, default=0.1, choices=[0.1, 0.2, 0.5])
    parser.add_argument("--CLIP", help="Gradient Clipping at", type=float, default=1, choices=[.1, 1, 10])
    parser.add_argument("--N_EPOCHS", help="Number of Epochs", type=int, default=100)
    parser.add_argument("--LEARNING_RATE", help="Learning Rate", type=float, default=0.0005, choices=[0.0005, 0.00005, 0.000005])
    args = parser.parse_args()

    SEED = 1234
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # df = pd.read_csv('./Dataset/sec_dataset_III_v3_masked_d1_gen.csv')
    df = pd.read_csv('./Dataset/detector_preds.csv')
    df['Error'] = df['Error'].apply(word2char)
    df['Word'] = df['Word'].apply(word2char)
    df['ErrorBlanksPredD1'] = df['ErrorBlanksPredD1'].apply(word2char)
    df['ErrorBlanksActual'] = df['ErrorBlanksActual'].apply(word2char)

    df['MaskErrorBlank'] = '<CLS> ' + df['Error'] + ' <SEP> ' + df['ErrorBlanksPredD1'] + ' <SEP>'
    df['Length'] = df['MaskErrorBlank'].apply(find_len)
    df = df.loc[df['Length'] <= 48]  # 48 works

    df = df.sample(frac=1).reset_index(drop=True)
    df = df[['ErrorBlanksActual', 'MaskErrorBlank', 'ErrorType']]

    train_df, valid_df, test_df = train_valid_test_df(df, test_size=0.15, valid_size=0.05)

    train_df.to_csv('./Dataset/train.csv', index=False)
    valid_df.to_csv('./Dataset/valid.csv', index=False)
    test_df.to_csv('./Dataset/test.csv', index=False)

    SRC = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>', batch_first=True
    )
    TRG = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>', batch_first=True
    )
    WORD = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>', batch_first=True
    )
    fields = {
        'MaskErrorBlank': ('src', SRC),
        'ErrorBlanksActual': ('trg', TRG)
    }

    train_data, valid_data, test_data = TabularDataset.splits(
        path='./Dataset',
        train='train.csv',
        validation='valid.csv',
        test='test.csv',
        format='csv',
        fields=fields
    )

    SRC.build_vocab(train_data, min_freq=100)  # 100
    TRG.build_vocab(train_data, min_freq=50)  # 50
    WORD.build_vocab(train_data, min_freq=100)  # 100

    # ------------------------------
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 512  # 512
    # ------------------------------
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    # ------------------------------
    HID_DIM = int(args.HID_DIM)
    ENC_LAYERS = int(args.ENC_LAYERS)
    DEC_LAYERS = int(args.DEC_LAYERS)
    ENC_HEADS = int(args.ENC_HEADS)
    DEC_HEADS = int(args.DEC_HEADS)
    ENC_PF_DIM = int(args.ENC_PF_DIM)
    DEC_PF_DIM = int(args.DEC_PF_DIM)
    ENC_DROPOUT = float(args.ENC_DROPOUT)
    DEC_DROPOUT = float(args.DEC_DROPOUT)
    CLIP = float(args.CLIP)
    N_EPOCHS = int(args.N_EPOCHS)
    LEARNING_RATE = float(args.LEARNING_RATE)
    # ------------------------------
    PATH = './Checkpoints/purificator.pth'
    # ------------------------------
    gc.collect()
    torch.cuda.empty_cache()
    # -----------------------------

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=DEVICE
    )

    enc = Encoder(
        INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM,
        ENC_DROPOUT, DEVICE
    )
    dec = Decoder(
        OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM,
        DEC_DROPOUT, DEVICE
    )

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, DEVICE).to(DEVICE)
    model.apply(initialize_weights)
    # print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    epoch = 1
    best_loss = 1e10
    if os.path.exists(PATH):
        checkpoint, epoch, train_loss = load_model(model, PATH)
        best_loss = train_loss

    for epoch in range(epoch, N_EPOCHS):
        print(f"Epoch: {epoch} / {N_EPOCHS}")
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        print(f"Train Loss: {train_loss:.4f}")
        if train_loss < best_loss:
            best_loss = train_loss
            save_model(model, train_loss, epoch, PATH)

    # ---------------------
    # eval_df = evaluation_report(test_data, SRC, TRG, model, DEVICE)
    # ---------------------
    # df = pd.read_csv('./Dataset/sec_dataset_III_v3_masked_d1_gen.csv')
    df = pd.read_csv('./Dataset/detector_preds.csv')
    # 
    error_types = list(set(df['ErrorType'].values))
    # 
    df['Error'] = df['Error'].apply(word2char)
    df['Word'] = df['Word'].apply(word2char)
    df['ErrorBlanksPredD1'] = df['ErrorBlanksPredD1'].apply(word2char)
    df['ErrorBlanksActual'] = df['ErrorBlanksActual'].apply(word2char)

    df['MaskErrorBlank'] = '<CLS> ' + df['Error'] + ' <SEP> ' + df['ErrorBlanksPredD1'] + ' <SEP>'
    df['Length'] = df['MaskErrorBlank'].apply(find_len)
    df = df.loc[df['Length'] <= 48]  # 48 works

    df = df.sample(frac=1).reset_index(drop=True)

    train_df, valid_df, test_df = train_valid_test_df2(df, test_size=1./1e10, valid_size=1./1e10)  # 1/1e10

    train_df.to_csv('./Dataset/train.csv', index=False)
    valid_df.to_csv('./Dataset/valid.csv', index=False)
    test_df.to_csv('./Dataset/test.csv', index=False)

    SRC = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>', batch_first=True
    )
    TRG = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>', batch_first=True
    )
    ERROR = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>', batch_first=True
    )
    WORD = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>', batch_first=True
    )
    EBPD1 = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>', batch_first=True
    )
    EBPFD1 = Field(
        tokenize=basic_tokenizer, lower=False,
        init_token='<sos>', eos_token='<eos>', batch_first=True
    )

    fields = {
        'MaskErrorBlank': ('src', SRC),
        'ErrorBlanksActual': ('trg', TRG),
        'Error': ('error', ERROR),
        'Word': ('word', WORD),
        'ErrorBlanksPredD1': ('ebpd1', EBPD1),
        'EBP_Flag_D1': ('ebpfd1', EBPFD1),
    }

    train_data, valid_data, test_data = TabularDataset.splits(
        path='./Dataset',
        train='train.csv',
        validation='valid.csv',
        test='test.csv',
        format='csv',
        fields=fields
    )

    SRC.build_vocab(train_data, min_freq=100)
    TRG.build_vocab(train_data, min_freq=50)
    ERROR.build_vocab(train_data, min_freq=100)
    WORD.build_vocab(train_data, min_freq=100)
    EBPD1.build_vocab(train_data, min_freq=100)
    EBPFD1.build_vocab(train_data, min_freq=100)

    # ---------------------
    for error_name in error_types:
        print(f'------\nError Type: {error_name}\n------')
        error_df_3(df, error_name)

        error_data, _ = TabularDataset.splits(
            path='./Dataset',
            train='error.csv',
            test='error.csv',
            format='csv',
            fields=fields
        )

        eval_df = evaluation_report3(
            error_data, SRC, TRG,
            ERROR, WORD, EBPD1, EBPFD1, model, DEVICE
        )
        eval_df['ErrorType'] = [error_name for _ in range(len(eval_df))]

        error_name = error_name.replace(' ', '').replace('(', '').replace(')', '')
        eval_df.to_csv(f'./Dataframes/purificator_{error_name}.csv', index=False)
        print('\n\n')
    # ---------------------
    merge_dfs(network='purificator')
    # ---------------------
    

if __name__ == '__main__':
    main()
