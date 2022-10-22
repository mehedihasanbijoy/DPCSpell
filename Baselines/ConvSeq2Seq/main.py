from utils import (
    basic_tokenizer, word2char, count_parameters, translate_sentence,
    save_model, load_model
)
from errors import error_df
from models import Encoder, Decoder, Seq2Seq
from pipeline import train, evaluate
from metrics import evaluation_report

import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import os
import argparse

import warnings as wrn
wrn.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CORPUS", help="Path of the Corpus", type=str, default="./Dataset/corpus2.csv", 
        choices=["./Dataset/corpus.csv", "./Dataset/corpus2.csv"]
    )
    parser.add_argument("--EMB_DIM", help="Embedding Dimension", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--HID_DIM", help="Hidden Dimension", type=int, default=256, choices=[64, 128, 256])
    parser.add_argument("--ENC_LAYERS", help="Encoder Layers", type=int,default=5, choices=[5, 10, 20])
    parser.add_argument("--DEC_LAYERS", help="Decoder Layers", type=int,default=5, choices=[5, 10, 20])
    parser.add_argument("--ENC_KERNEL_SIZE", help="Encoder Kernel Size", type=int, default=3, choices=[3, 5, 10])
    parser.add_argument("--DEC_KERNEL_SIZE", help="Decoder Kernel Size", type=int, default=3, choices=[3, 5, 10])
    parser.add_argument("--ENC_DROPOUT", help="Encoder Dropout Ratio", type=float, default=.2, choices=[.1, .2, .5])
    parser.add_argument("--DEC_DROPOUT", help="Decoder Dropout Ratio", type=float, default=.2, choices=[.1, .2, .5])
    parser.add_argument("--CLIP", help="Gradient Clipping", type=float, default=0.1, choices=[0.1, 0.2, 0.5, 1])
    parser.add_argument("--BATCH_SIZE", help="Batch Size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--N_EPOCHS", help="Number of Epochs", type=int, default=100)
    parser.add_argument("--LEARNING_RATE", help="Learning Rate", type=float, default=0.0005, choices=[0.0005, 0.00005, 0.000005])
    args = parser.parse_args()

    df = pd.read_csv(args.CORPUS)
    df['Word'] = df['Word'].apply(word2char)
    df['Error'] = df['Error'].apply(word2char)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[['Error', 'Word']]

    train_df, test_df = train_test_split(df, test_size=.15)
    train_df, valid_df = train_test_split(train_df, test_size=.05)

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

    SRC.build_vocab(train_data, min_freq=100)
    TRG.build_vocab(train_data, min_freq=50)

    # Hyperparameters
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = args.BATCH_SIZE
    #
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    EMB_DIM = args.EMB_DIM  # 64
    HID_DIM = args.HID_DIM  # 256 # each conv. layer has 2 * hid_dim filters
    ENC_LAYERS = args.ENC_LAYERS  # 10  # number of conv. blocks in encoder
    DEC_LAYERS = args.DEC_LAYERS  # 10  # number of conv. blocks in decoder
    ENC_KERNEL_SIZE = args.ENC_KERNEL_SIZE  # must be odd!
    DEC_KERNEL_SIZE = args.DEC_KERNEL_SIZE  # can be even or odd
    ENC_DROPOUT = args.ENC_DROPOUT
    DEC_DROPOUT = args.DEC_DROPOUT
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    CLIP = args.CLIP
    PATH = './Checkpoints/conv_s2s.pth'

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=DEVICE
    )

    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, DEVICE)
    dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, DEVICE)
    model = Seq2Seq(enc, dec).to(DEVICE)
    # print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    epoch = 1
    # load the model
    if os.path.exists(PATH):
        checkpoint, epoch, train_loss = load_model(model, PATH)
    #
    best_loss = 1e10

    for epoch in range(epoch, N_EPOCHS):
        print(f"Epoch: {epoch} / {N_EPOCHS}")
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        print(f"Train Loss: {train_loss:.4f}")
        if train_loss < best_loss:
            best_loss = train_loss
            save_model(model, train_loss, epoch, PATH)

    # example_idx = 10
    # src = vars(train_data.examples[example_idx])['src']
    # trg = vars(train_data.examples[example_idx])['trg']
    # print(f'src = {src}')
    # print(f'trg = {trg}')
    # translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)
    # print(f'predicted trg = {translation}')

    evaluation_report(valid_data, SRC, TRG, model, DEVICE)
    # evaluation_report(error_data, SRC, TRG, model, DEVICE)


    # -------------
    # error_types = ['Cognitive Error', 'Homonym Error', 'Run-on Error',
    #  'Split-word Error (Left)', 'Split-word Error (Random)',
    #  'Split-word Error (Right)', 'Split-word Error (both)',
    #  'Typo (Avro) Substituition', 'Typo (Bijoy) Substituition',
    #  'Typo Deletion', 'Typo Insertion', 'Typo Transposition',
    #  'Visual Error', 'Visual Error (Combined Character)']

    # for error_name in error_types:
    #     print(f'------\nError Type: {error_name}\n------')
    #     error_df(df_copy, error_name)

    #     error_data, _ = TabularDataset.splits(
    #         path='./Dataset',
    #         train='error.csv',
    #         test='error.csv',
    #         format='csv',
    #         fields=fields
    #     )

    #     eval_df = evaluation_report(error_data, SRC, TRG, model, DEVICE)

    #     error_name = error_name.replace(' ', '').replace('(', '').replace(')', '')
    #     eval_df.to_csv(f'./Dataframes/convs2s_{error_name}_2.csv')
    #     print('\n\n')
    # -------------


if __name__ == '__main__':
    main()
