import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, device, max_length=50):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size should be odd in encoder"
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hid_dim, out_channels= 2 *hid_dim, kernel_size=kernel_size, padding=(kernel_size-1 )//2
            ) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout(tok_embedded + pos_embedded)
        conv_inp = self.emb2hid(embedded)
        conv_inp = conv_inp.permute(0, 2, 1)

        for idx, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_inp))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_inp) * self.scale
            conv_inp = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))
        combined = (conved + embedded) * self.scale
        return conved, combined


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, \
                 trg_pad_idx, device, max_length=50):
        super().__init__()
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hid_dim, out_channels=2 * hid_dim, kernel_size=kernel_size
            ) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
        combined = (conved_emb + embedded) * self.scale
        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        attention = F.softmax(energy, dim=2)
        attended_encoding = torch.matmul(attention, encoder_combined)
        attended_encoding = self.attn_emb2hid(attended_encoding)
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale
        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout(tok_embedded + pos_embedded)

        conv_inp = self.emb2hid(embedded)
        conv_inp = conv_inp.permute(0, 2, 1)

        batch_size = conv_inp.shape[0]
        hid_dim = conv_inp.shape[1]
        for idx, conv in enumerate(self.convs):
            conv_inp = self.dropout(conv_inp)
            padding = torch.zeros(
                batch_size, hid_dim, self.kernel_size - 1
            ).fill_(self.trg_pad_idx).to(self.device)
            padded_conv_inp = torch.cat((padding, conv_inp), dim=2)
            conved = conv(padded_conv_inp)
            conved = F.glu(conved, dim=1)

            attention, conved = self.calculate_attention(
                embedded, conved, encoder_conved, encoder_combined
            )
            conved = (conved + conv_inp) * self.scale
            conv_inp = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))
        output = self.fc_out(self.dropout(conved))
        return output, attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        encoder_conved, encoder_combined = self.encoder(src)
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)
        return output, attention

if __name__ == '__main__':
    pass