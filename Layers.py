import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        # temperature 应该就是除的那个缩放因子，防止点积之后结果处于softmax函数梯度很小的区域
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # 这里的mask是padding mask，因为每次输入的数据不一样长，为了对齐。不空为0，空白为1

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head  # 头数
        self.d_k = d_k  # Q、K、V的维度
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, self.n_head * d_k)
        self.w_ks = nn.Linear(d_model, self.n_head * d_k)
        self.w_vs = nn.Linear(d_model, self.n_head * d_v)
        # 不知道为什么没办法初始化
        # nn.init.normal(self.w_qs, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_ks, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_vs, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        # multi-head attention需要layer norm
        self.attention = ScaledDotProductAttention(dropout)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)  # 还是layer_norm

    def forward(self, q, k, v, mask=None):
        # d_k， d_v，n_head分别是Q，K，V的维度，以及Attention的头数
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q  # 用于残差连接

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b x lq x dk)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b x lk x dk)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b x lv x dv)

        mask = mask.repeat(n_head, 1, 1)  # (n*b , bz, x)
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)  # 做残差连接，在反向传播中，即使梯度连乘，也不会造成梯度消失
        # 同时做 layer_norm，类似与batch normalization，只不过，这是针对每个样本，而不是每个batch
        # 经过神经网络后，数据偏差很大，为了防止梯度消失或梯度爆炸

        return output, attn

class PositiowiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositiowiseFeedForward, self).__init__()

        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)  # layer norm，不同于batch norm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # encoder 层输入的x
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # 多头的注意力机制
        self.slf_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=dropout)
        #
        self.pos_ffn = PositiowiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        # 输入包括Q，K，V三个矩阵的维度，这个mask可以暂时不用
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)

        # enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        # enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

# 生成位置编码
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusodi_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusodi_table[:, 0::2] = np.sin(sinusodi_table[:, 0::2])  # 偶数索引
    sinusodi_table[:, 1::2] = np.cos(sinusodi_table[:, 1::2])  # 奇数索引

    # 暂时略过padding_idx

    return torch.FloatTensor(sinusodi_table)

# 这个mask暂时不明白
def get_attn_key_pad_mask(seq_k, seq_q):

    len_q = seq_q.size(1)
    print(seq_k)
    padding_mask = seq_k.eq()  # 没有加PAD
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

# 这个mask也是不很明白
def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)

class Encoder(nn.Module):

    def __init__(self, n_src_vocab, len_max_seq, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super(Encoder, self).__init__()

        n_position = len_max_seq + 1

        # 原始词嵌入
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec)  # 没有加PAD,d_word_vec是嵌入维度，单词的个数

        # 位置编码
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec), freeze=True)  # freeze应该是固定的意思，不释放

        # 多层encoder
        self.layer_stack = nn.ModuleList(
            [EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)  # 添加mask
        # non_pad_mask = get_non_pad_mask(src_seq)  # 添加mask
        print(src_seq.size())
        print(src_seq)
        word_embedding = self.src_word_emb(src_seq)
        enc_output = word_embedding + self.position_enc(src_pos)  # 生成词语embedding
                                                                            # word embedding+position embedding

        for enc_layer in self.layer_stack:  # n_layers的encoder layer，循环n_layers次经过encoder layers
            enc_output, enc_slf_attn = enc_layer( enc_output)
                # enc_output, non_pad_mask, slf_attn_mask)
            if return_attns:  # 是否要返回Attention参数
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


