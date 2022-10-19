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

import warnings as wrn
wrn.filterwarnings('ignore')


def word2chars(word):
    w2c = [char for char in word]
    return ' '.join(w2c)


def df2train_test_dfs(df, test_size=0.15):
    df['Word'] = df['Word'].apply(word2chars)
    df['Error'] = df['Error'].apply(word2chars)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.iloc[:, [1, 0]]
    train_df, test_df = train_test_split(df, test_size=test_size)
    train_df.to_csv('./Dataset/train.csv', index=False)
    test_df.to_csv('./Dataset/test.csv', index=False)


def df2train_valid_test_dfs(df, test_size=0.15):
    df['Word'] = df['Word'].apply(word2chars)
    df['Error'] = df['Error'].apply(word2chars)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.iloc[:, [1, 0]]
    train_df, test_df = train_test_split(df, test_size=test_size)
    train_df, valid_df = train_test_split(train_df, test_size=.05)

    train_df.to_csv('./Dataset/train.csv', index=False)
    valid_df.to_csv('./Dataset/valid.csv', index=False)
    test_df.to_csv('./Dataset/test.csv', index=False)


def df2train_error_dfs(df, error='Cognitive Error', test_size=0.20):
    df['Word'] = df['Word'].apply(word2chars)
    df['Error'] = df['Error'].apply(word2chars)
    df = df.sample(frac=1).reset_index(drop=True)
    # df = df.iloc[:, [1, 0]]
    train_df, error_df = train_test_split(df, test_size=test_size)
    error_df = error_df.loc[error_df['ErrorType'] == error]
    train_df = train_df.iloc[:, [1, 0]]
    error_df = error_df.iloc[:, [1, 0]]

    train_df.to_csv('./Dataset/train.csv', index=False)
    error_df.to_csv('./Dataset/error.csv', index=False)


def basic_tokenizer(text):
    return text.split()


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, epoch, optimizer, train_loss, PATH):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss
    }, PATH)
    print(f"---------\nModel Saved at {PATH}\n---------\n")


def load_model(model, optimizer, PATH):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['loss']
    return checkpoint, epoch, train_loss


def print_n_best(decoded_seq, itos):
    topk_preds = []
    for rank, seq in enumerate(decoded_seq):
        pred = "".join([itos[idx] for idx in seq[1:-1]])
        topk_preds.append(pred)
        # print(f'Out: Rank-{rank+1}: {pred}')
    return topk_preds


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=30):
    model.eval()
    tokens = [token for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(src_indexes)])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attentions[:len(trg_tokens) - 1]


def display_attention(sentence, translation, attention):
    prop = fm.FontProperties(fname='./Dataset/kalpurush.ttf')

    fig = plt.figure(figsize=(7, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)

    x_ticks = [''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>']
    y_ticks = [''] + translation

    ax.set_xticklabels(x_ticks, rotation=0, fontproperties=prop, fontsize=20)
    ax.set_yticklabels(y_ticks, fontproperties=prop, fontsize=20)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()
