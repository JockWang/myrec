import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TransEncoder

class TransMdl(nn.Module):
    def __init__(self, d_model, n_layers, n_head, seq_len, d_inner, dropout=0.):
        super(TransMdl, self).__init__()

        self.encoder = TransEncoder(n_layers=n_layers, n_head=n_head, seq_len=seq_len,
                                    d_model=d_model, d_inner=d_inner, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.active = nn.ReLU()

    def forward(self, src_seq):
        out = self.active(self.encoder(src_seq))

        return self.dropout(out)

class Model(nn.Module):
    def __init__(self, qs_input_dim, as_input_dim, mb_dim, hidden_units=None, mb_units=None, n_layers=6, n_head=2,
                 seq_len=9, d_inner=512, init_std=0.0001, activation=F.relu, dropout_rate=0, device='cpu'):
        super(Model, self).__init__()
        self.qes = TransMdl(qs_input_dim, n_layers, n_head, seq_len, d_inner, dropout_rate)
        self.ans = TransMdl(as_input_dim, n_layers, n_head, seq_len, d_inner, dropout_rate)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

        mb_units = [mb_dim] + list(mb_units)
        self.mb_linears = nn.ModuleList([nn.Linear(mb_units[i], mb_units[i+1])
                                        for i in range(len(mb_units)-1)])
        for name, tensor in self.mb_linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        if len(hidden_units) == 0:
            raise ValueError('Hidden units is empty!!!')
        hidden_units = [mb_units[-1]+qs_input_dim+as_input_dim+qs_input_dim] + list(hidden_units)
        self.linears = nn.ModuleList([nn.Linear(hidden_units[i], hidden_units[i+1])
                                      for i in range(len(hidden_units)-1)])
        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, qs_feat, as_feat, invite, member):

        qes_out = self.qes(qs_feat)
        ans_out = self.ans(as_feat)
        meb_out = member
        for i in range(len(self.mb_linears)):
            meb_out = self.mb_linears[i](meb_out)
            meb_out = self.activation(meb_out)
            meb_out = self.dropout(meb_out)

        out = torch.cat((meb_out, qes_out, ans_out, invite), 1)

        for i in range(len(self.linears)):
            out = self.linears[i](out)
            out = self.activation(out)
            out = self.dropout(out)

        return out
