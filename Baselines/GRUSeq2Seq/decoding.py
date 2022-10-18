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

import copy
from heapq import heappush, heappop

import warnings as wrn
wrn.filterwarnings('ignore')


class BeamSearchNode(object):
    def __init__(self, h, prev_node, wid, logp, length):
        self.h = h
        self.prev_node = prev_node
        self.wid = wid
        self.logp = logp
        self.length = length

    def eval(self):
        return self.logp / float(self.length - 1 + 1e-6)


def beam_search_decoding(model, src, decoder, enc_outs, enc_last_h, beam_width, n_best, \
                         sos_token, eos_token, max_dec_steps, device):
    assert beam_width >= n_best
    n_best_list = []
    bs = enc_outs.shape[1]

    for batch_id in range(bs):
        decoder_hidden = enc_last_h[batch_id]
        enc_out = enc_outs[:, batch_id].unsqueeze(1)

        # decoder_input = torch.tensor([sos_token].long().to(DEVICE))
        decoder_input = torch.tensor([sos_token]).to(device)
        end_nodes = []

        node = BeamSearchNode(h=decoder_hidden, prev_node=None, wid=decoder_input, logp=0, length=1)
        nodes = []

        heappush(nodes, (-node.eval(), id(node), node))
        n_dec_steps = 0

        while True:
            if n_dec_steps > max_dec_steps:
                break

            score, _, n = heappop(nodes)
            decoder_input = n.wid
            decoder_hidden = n.h

            if n.wid.item() == eos_token and n.prev_node is not None:
                end_nodes.append((score, id(n), n))
                if len(end_nodes) >= n_best:
                    break
                else:
                    continue

            mask = model.create_mask(src)
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden.unsqueeze(0), enc_out, mask)

            # restricting length
            topk_log_prob, topk_indexes = torch.topk(decoder_output, beam_width)

            for new_k in range(beam_width):
                decoded_t = topk_indexes[0][new_k].view(1)
                logp = topk_log_prob[0][new_k].item()

                node = BeamSearchNode(
                    h=decoder_hidden.squeeze(0), prev_node=n, wid=decoded_t, logp=n.logp + logp, length=n.length + 1
                )

                heappush(nodes, (-node.eval(), id(node), node))

            n_dec_steps += beam_width

        if len(end_nodes) == 0:
            end_nodes = [heappop(nodes) for _ in range(beam_width)]

        n_best_seq_list = []
        for score, _id, n in sorted(end_nodes, key=lambda x: x[0]):
            sequence = [n.wid.item()]
            while n.prev_node is not None:
                n = n.prev_node
                sequence.append(n.wid.item())
            sequence = sequence[::-1]
            n_best_seq_list.append(sequence)

        n_best_list.append(n_best_seq_list)

    return n_best_list






