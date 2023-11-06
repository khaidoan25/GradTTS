import torch
from torch import nn

from models import DurationPredictor
from style_speech import Constants
from text.symbols import symbols
from style_speech.modules import Mish, LinearNorm, Conv1dGLU, ConvNorm, \
    MultiHeadAttention, StyleAdaptiveLayerNorm, get_sinusoid_encoding_table


class MelStyleEncoder(nn.Module):
    ''' MelStyleEncoder '''
    def __init__(self, config):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = config.n_mel_channels 
        self.hidden_dim = config.style_hidden
        self.out_dim = config.style_vector_dim
        self.kernel_size = config.style_kernel_size
        self.n_head = config.style_head
        self.dropout = config.dropout

        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout)
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = MultiHeadAttention(self.n_head, self.hidden_dim, 
                                self.hidden_dim//self.n_head, self.hidden_dim//self.n_head, self.dropout) 

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, x, mask=None):
        max_len = x.shape[1] # b, t, h
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        
        # spectral
        x = self.spectral(x)
        # temporal
        x = x.transpose(1,2)
        x = self.temporal(x)
        x = x.transpose(1,2)
        # self-attention
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)

        return w
    
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, fft_conv1d_kernel_size, dropout=0.1):
        super().__init__()
        self.w_1 = ConvNorm(d_in, d_hid, kernel_size=fft_conv1d_kernel_size[0])
        self.w_2 =  ConvNorm(d_hid, d_in, kernel_size=fft_conv1d_kernel_size[1])

        self.mish = Mish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        residual = input

        output = input.transpose(1, 2)
        output = self.w_2(self.dropout(self.mish(self.w_1(output))))
        output = output.transpose(1, 2)

        output = self.dropout(output) + residual
        return output
    
    
class Prenet(nn.Module):
    ''' Prenet '''
    def __init__(self, hidden_dim, out_dim, dropout):
        super(Prenet, self).__init__()

        self.convs = nn.Sequential(
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3),
            Mish(),
            nn.Dropout(dropout),
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3),
            Mish(),
            nn.Dropout(dropout),
        )
        self.fc = LinearNorm(hidden_dim, out_dim)

    def forward(self, input, mask=None):
        # residual = input
        # convs
        output = input.transpose(1,2)
        output = self.convs(output)
        output = output.transpose(1,2)
        # fc & residual
        output = self.fc(output)

        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)
        return output
    
    
class FFTBlock(nn.Module):
    ''' FFT Block '''
    def __init__(self, d_model,d_inner,
                    n_head, d_k, d_v, fft_conv1d_kernel_size, style_dim, dropout):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.saln_0 = StyleAdaptiveLayerNorm(d_model, style_dim)

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel_size, dropout=dropout)
        self.saln_1 = StyleAdaptiveLayerNorm(d_model, style_dim)

    def forward(self, input, style_vector, mask=None, slf_attn_mask=None):
        # multi-head self attn
        slf_attn_output, slf_attn = self.slf_attn(input, mask=slf_attn_mask)
        slf_attn_output = self.saln_0(slf_attn_output, style_vector)
        if mask is not None:
            slf_attn_output = slf_attn_output.masked_fill(mask.unsqueeze(-1), 0)

        # position wise FF
        output = self.pos_ffn(slf_attn_output)
        output = self.saln_1(output, style_vector)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        return output, slf_attn
    
    
class StyleAdaptiveEncoder(nn.Module):
    def __init__(self, config):
        super(StyleAdaptiveEncoder, self).__init__()
        
        self.n_layers = config.encoder_layer
        self.d_model = config.encoder_hidden
        self.n_head = config.encoder_head
        self.d_k = config.encoder_hidden // config.encoder_head
        self.d_v = config.encoder_hidden // config.encoder_head
        self.d_inner = config.fft_conv1d_filter_size
        self.fft_conv1d_kernel_size = config.fft_conv1d_kernel_size
        self.style_dim = config.style_vector_dim
        self.dropout = config.dropout
        self.n_mel_channels = config.n_mel_channels
        
        self.prenet = Prenet(self.n_mel_channels, self.d_model, self.dropout)
        
        self.layer_stack = nn.ModuleList([FFTBlock(
            self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, 
            self.fft_conv1d_kernel_size, self.style_dim, self.dropout) for _ in range(self.n_layers)])
        
        self.postnet = Prenet(self.d_model, self.n_mel_channels, self.dropout)
        
    def forward(self, x, style_vector):
        x = self.prenet(x)
        for enc_layer in self.layer_stack:
            x, _ = enc_layer(x, style_vector)
        x = self.postnet(x)
        
        return x