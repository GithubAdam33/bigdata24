import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.dropout
        )
        # Encoder
        self.expert_1 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.expert_2 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.expert_3 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # Decoder
        # p = 0.1 # dropout rate
        # self.wind_gate = nn.Sequential(nn.Linear(configs.d_model, 3, bias=True), nn.Softmax(dim = -1), nn.Dropout(p))
        # self.temp_gate = nn.Sequential(nn.Linear(configs.d_model, 3, bias=True), nn.Softmax(dim = -1), nn.Dropout(p))
        self.wind_gate = nn.Linear(configs.d_model, 3, bias=True)
        self.temp_gate = nn.Linear(configs.d_model, 3, bias=True)

        # d_model = 64
        self.wind_projection = nn.Linear(128, configs.pred_len, bias=True)
        self.temp_projection = nn.Linear(128, configs.pred_len, bias=True)
        
        # self.wind_projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        # self.temp_projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.sf1 = nn.Softmax(dim=-1)
        self.sf2 = nn.Softmax(dim=-1)

        # self.lstm_w = nn.LSTM(
        #     configs.d_model, 128, batch_first=True, dropout=configs.dropout
        # )
        # self.lstm_t = nn.LSTM(
        #     configs.d_model, 128, batch_first=True, dropout=configs.dropout
        # )

        self.bilstm_w = nn.LSTM(
            configs.d_model, 64, batch_first=True, dropout=configs.dropout, bidirectional=True
        )
        self.bilstm_t = nn.LSTM(
            configs.d_model, 64, batch_first=True, dropout=configs.dropout, bidirectional=True
        )
        
        # self.lstm_w = nn.LSTM(
        #     configs.enc_in, 64, batch_first=True, dropout=configs.dropout
        # )
        # self.lstm_t = nn.LSTM(
        #     configs.enc_in, 64, batch_first=True, dropout=configs.dropout
        # )
        # self.out_w = nn.Sequential(
        #     nn.Linear(64, configs.enc_in, bias=True),
        # )
        # self.out_t = nn.Sequential(
        #     nn.Linear(64, configs.enc_in, bias=True),
        # )

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)  # (bs,38,64)

        enc_out_1, attns_1 = self.expert_1(
            enc_out, attn_mask=None
        )  # batchsize inputsize seq_L
        enc_out_2, attns_2 = self.expert_2(enc_out, attn_mask=None)
        enc_out_3, attns_3 = self.expert_3(enc_out, attn_mask=None)

        enc_out_e = torch.stack([enc_out_1, enc_out_2, enc_out_3], dim=3)

        # wg = self.wind_gate(enc_out).unsqueeze(2)
        wg = self.wind_gate(enc_out)
        wg = self.sf1(wg).unsqueeze(2)
        wg = wg.expand_as(enc_out_e)

        # tg = self.temp_gate(enc_out).unsqueeze(2)
        tg = self.temp_gate(enc_out)
        tg = self.sf1(tg).unsqueeze(2)
        tg = tg.expand_as(enc_out_e)

        wg = (enc_out_e * wg).sum(dim=3)  # (bs,38,64)
        tg = (enc_out_e * tg).sum(dim=3)

        # # LSTM
        # wg, _ = self.lstm_w(wg)
        # tg, _ = self.lstm_t(tg)
        
        # BiLSTM
        wg, _ = self.bilstm_w(wg)
        tg, _ = self.bilstm_t(tg)
        
        # projection
        wg = self.wind_projection(wg).permute(0, 2, 1)[:, :, :N]  # (bs,24,38)
        tg = self.temp_projection(tg).permute(0, 2, 1)[:, :, :N]  # (bs,24,38)

        # # projection
        # wg = self.wind_projection(wg).permute(0, 2, 1)
        # tg = self.temp_projection(tg).permute(0, 2, 1)
        # # LSTM
        # wg, _ = self.lstm_w(wg)
        # tg, _ = self.lstm_t(tg)
        # # projection
        # wg = self.out_w(wg)
        # tg = self.out_t(tg)

        # De-Normalization from Non-stationary Transformer
        wg = wg * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        wg = wg + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        tg = tg * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        tg = tg + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return wg, tg

    def forward(self, x_enc):
        wind, temp = self.forecast(x_enc)
        return wind[:, -self.pred_len :, :], temp[:, -self.pred_len :, :]  # [B, L, C]
