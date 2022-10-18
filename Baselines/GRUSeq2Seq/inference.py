from decoding import beam_search_decoding
from metrics import beam_eval_report, greedy_eval_report
from utils import print_n_best
from utils import translate_sentence

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


def test_beam(model, train_data, test_data, SRC, TRG, DEVICE):
    _, test_iterator = BucketIterator.splits(
        (train_data, test_data),
        batch_size=1,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=DEVICE
    )

    TRG_SOS_IDX = TRG.vocab.stoi[TRG.init_token]
    TRG_EOS_IDX = TRG.vocab.stoi[TRG.eos_token]

    src_words = []
    topk_prediction_list = []
    trg_words = []
    found_within_topk = []
    found_at_top1 = []

    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(test_iterator)):
            src, src_len = batch.src
            trg = batch.trg

            src_word = "".join(SRC.vocab.itos[idx] for idx in src[:, 0][1:-1])
            trg_word = "".join(TRG.vocab.itos[idx] for idx in trg[:, 0][1:-1])
            # print(f'\nSRC: {src_word}')
            # print(f'\nTRG: {trg_word}')

            enc_outs, h = model.encoder(src, src_len)
            # print(enc_outs.shape, h.shape)

            # decoder, enc_outs, enc_last_h, beam_width, n_best, sos_token, eos_token, max_dec_steps, device
            decoded_seqs = beam_search_decoding(
                model = model,
                src = src,
                decoder=model.decoder,
                enc_outs=enc_outs,
                enc_last_h=h,
                beam_width=1,
                n_best=1,
                sos_token=TRG_SOS_IDX,
                eos_token=TRG_EOS_IDX,
                max_dec_steps=100,
                device=DEVICE
            )
            topk_preds = print_n_best(decoded_seqs[0], TRG.vocab.itos)
            # print(topk_preds)

            src_words.append(src_word)
            trg_words.append(trg_word)
            topk_prediction_list.append((topk_preds * 3)[:3])
            found_within_topk.append(1) if trg_word in topk_preds else found_within_topk.append(0)
            found_at_top1.append(1) if trg_word == topk_preds[0] else found_at_top1.append(0)

            # if batch_id == 100:
            #     break

    topk_pred_df = pd.DataFrame({
        'Error': src_words,
        'Pred-1': np.array(topk_prediction_list)[:, 0],
        'Pred-2': np.array(topk_prediction_list)[:, 1],
        'Pred-3': np.array(topk_prediction_list)[:, 2],
        'Correct': trg_words,
        'Greedy': found_at_top1,
        'Beam': found_within_topk
    })
    topk_pred_df.to_csv('./Corrections/preds_beam.csv', index=False)

    beam_eval_report(trg_words, topk_prediction_list)


def test_greedy(test_data, SRC, TRG, model, DEVICE):
    erroneous_words, predicted_words, correct_words, flags = [], [], [], []
    for idx, data in enumerate(tqdm(test_data)):
        src = data.src
        trg = data.trg
        translation, attention = translate_sentence(src, SRC, TRG, model, DEVICE)

        src = ''.join(src)
        trg = ''.join(trg)
        pred = ''.join(translation[:-1])

        erroneous_words.append(src)
        predicted_words.append(pred)
        correct_words.append(trg)
        if trg == pred:
            flags.append(1)
        else:
            flags.append(0)

    evaluation_df = pd.DataFrame({
        'Error': erroneous_words,
        'Predicton': predicted_words,
        'Target': correct_words,
        'Correction': flags
    })
    evaluation_df.to_csv('./Corrections/preds_greedy.csv', index=False)

    greedy_eval_report(correct_words, predicted_words)


