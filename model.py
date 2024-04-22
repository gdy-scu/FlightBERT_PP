# coding=utf-8
import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("mask", torch.tril(
            torch.ones(config.inp_seq_len + config.horizon, config.inp_seq_len + config.horizon))
                             .view(1, 1, config.inp_seq_len + config.horizon, config.inp_seq_len + config.horizon))
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.mlp(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Feature_Aggregation(nn.Module):
    def __init__(self, hidden_dim):
        super(Feature_Aggregation, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 8),
                                        nn.GELU(),
                                        nn.Linear(hidden_dim // 8, 1)
                                        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        ouputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return ouputs  # , weights


class DecTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(DecTokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1,
                                   padding_mode='zeros')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class EncTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(EncTokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1,
                                   padding_mode='zeros')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FlightBERT_PP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.alt_size = config.alt_size
        self.spdx_size = config.spdx_size
        self.spdy_size = config.spdy_size
        self.spdz_size = config.spdz_size

        self.full_size = config.delta_lon_size + config.delta_lat_size + config.delta_alt_size + config.delta_spdx_size + config.delta_spdy_size + config.delta_spdz_size

        self.lat_embd = config.lat_embed
        self.lon_embd = config.lon_embed
        self.alt_embd = config.alt_embed
        self.spdx_embd = config.spdx_embed
        self.spdy_embd = config.spdy_embed
        self.spdz_embd = config.spdz_embed

        self.n_embd = config.lat_embed + config.lon_embed + config.alt_embed + config.spdx_embed + config.spdy_embed + config.spdz_embed
        in_dim = self.lat_size + self.lon_size + self.alt_size + self.spdx_size + self.spdy_size + self.spdz_size
        self.enc_embed = EncTokenEmbedding(in_dim, self.n_embd)

        self.pos_emb = PositionalEncoding(self.config.n_embd, dropout=0.1, max_len=config.inp_seq_len)
        self.de_pos_emb = PositionalEncoding(self.config.n_embd, dropout=0.1,
                                             max_len=config.inp_seq_len + config.horizon)

        # transformer encoder
        encoder_layers = TransformerEncoderLayer(config.n_embd, config.n_head, config.n_embd, config.encoder_drop)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.n_en_layer)

        # self.transformer_encoder = nn.Sequential(*[Block(config) for _ in range(config.n_en_layer)])

        self.feature_aggre = Feature_Aggregation(config.n_embd)

        self.dec_token_emb = DecTokenEmbedding(self.full_size, self.config.n_embd)

        self.horizon_emb = nn.Embedding(num_embeddings=config.horizon, embedding_dim=config.n_embd)
        self.horizon_linear = nn.Linear(config.n_embd * 2, config.n_embd)

        # decoder_layers = TransformerDecoderLayer(config.n_embd, config.n_head, config.n_embd, config.encoder_drop)
        # self.transformer_decoder = TransformerDecoder(decoder_layers, config.n_de_layer)

        self.decoder = nn.Sequential(*[Block(config) for _ in range(config.n_de_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, self.full_size, bias=False)  # Classification head

        self.inp_seqlen = config.inp_seq_len
        # self.apply(self._init_weights)
        self.sigmod = nn.Sigmoid()

        self.horizons = torch.arange(0, self.config.horizon).unsqueeze(dim=0).repeat(self.config.batch_size, 1)

        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_max_seqlen(self):
        return self.inp_seqlen

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, lon_inp, lat_inp, alt_inp, spdx_inp, spdy_inp, spdz_inp, dec_inputs):
        cat_inputs = torch.cat((lon_inp, lat_inp, alt_inp, spdx_inp, spdy_inp, spdz_inp), dim=-1)
        token_embeddings = self.enc_embed(cat_inputs)

        token_embeddings = self.pos_emb(token_embeddings.transpose(1, 0)).transpose(1, 0)

        encoder_output = self.transformer_encoder(token_embeddings)

        context_embed = self.feature_aggre(encoder_output)

        device = context_embed.device

        if context_embed.shape[0] != self.config.batch_size:
            self.horizons = torch.arange(0, self.config.horizon).unsqueeze(dim=0).repeat(context_embed.shape[0], 1)

        if self.horizons.shape[0] != context_embed.shape[0]:
            self.horizons = torch.arange(0, self.config.horizon).unsqueeze(dim=0).repeat(context_embed.shape[0], 1)

        horizon_embed = self.horizon_emb(self.horizons.long().to(device))

        context_embed = context_embed.unsqueeze(dim=1).repeat(1, self.config.horizon, 1)
        horizon_context = torch.cat([context_embed, horizon_embed], dim=2)

        decoder_input = self.horizon_linear(horizon_context)

        decoder_input_diff = self.dec_token_emb(dec_inputs)
        decoder_input = torch.cat((decoder_input_diff, decoder_input), dim=1)

        decoder_input = self.de_pos_emb(decoder_input.transpose(1, 0)).transpose(1, 0)

        fea = self.decoder(decoder_input)

        fea = self.ln_f(fea)
        logits = self.head(fea)

        logits = self.sigmod(logits)[:, -self.config.horizon:, :]

        lon_logits, lat_logits, alt_logits, spdx_logits, spdy_logits, spdz_logits = \
            torch.split(logits, (self.config.delta_lon_size, self.config.delta_lat_size, self.config.delta_alt_size,
                                 self.config.delta_spdx_size, self.config.delta_spdy_size,
                                 self.config.delta_spdz_size), dim=-1)

        return lon_logits, lat_logits, alt_logits, spdx_logits, spdy_logits, spdz_logits
