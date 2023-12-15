import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import pickle
import os
import random
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch_geometric.nn import SAGEConv, GATConv, ChebConv, TransformerConv
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.utils import unbatch
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from typing import Optional, Tuple
# from torch.nn.modules.activation import MultiheadAttention
from torch.nn import Linear, Dropout
from torch.nn import LayerNorm
from torch.nn.modules.transformer import _get_activation_fn, _get_clones
from torch_geometric.nn.pool import global_mean_pool


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2023)


class NodeEmbeding(nn.Module):

    def __init__(self, nodeid2token, d_model=128):
        super(NodeEmbeding, self).__init__()
        node2vec = pickle.load(
            open('../dataset/node_embedding_longhua_1.8k.pkl',
                 'rb'))
        self.node_embedding = nn.Embedding(nodeid2token.vocab_size, d_model)
        # self.embedding_weights(np.stack(used_records.car_feature.values))
        self.embedding_weights(node2vec)

    def embedding_weights(self, node_emb, records_weights=None):
        self.node_embedding.weight.data[-node_emb.shape[0]:].copy_(
            torch.from_numpy(node_emb))
        self.node_embedding.weight.data.requires_grad = False

    def forward(self, x):
        return self.node_embedding(x).detach()


class TemporalEncoding(torch.nn.Module):
    r"""The time-encoding function from the `"Do We Really Need Complicated
    Model Architectures for Temporal Networks?"
    <https://openreview.net/forum?id=ayPPc0SyLv1>`_ paper.
    :class:`TemporalEncoding` first maps each entry to a vector with
    monotonically exponentially decreasing values, and then uses the cosine
    function to project all values to range :math:`[-1, 1]`

    .. math::
        y_{i} = \cos \left(x \cdot \sqrt{d}^{-(i - 1)/\sqrt{d}} \right)

    where :math:`d` defines the output feature dimension, and
    :math:`1 \leq i \leq d`.

    Args:
        out_channels (int): Size :math:`d` of each output sample.
    """
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        sqrt = math.sqrt(out_channels)
        weight = 1.0 / sqrt**torch.linspace(0, sqrt, out_channels).view(1, -1)
        self.register_buffer('weight', weight)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        """"""
        batch_size, seq_len = x.size()
        return torch.cos(x.reshape(-1, 1) @ self.weight).reshape(batch_size, seq_len, -1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'


class PositionEncoding(nn.Module):
    """
    position embedding in transformer
    """

    def __init__(self, d_model, max_len=200):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)        

    def forward(self, x):
        """
        :param x:
        :return: embedding
        """
        return x + self.pe[:, :x.size(1)]


class TransModel(nn.Module):

    def __init__(self):
        super(TransModel, self).__init__()

    def _sequence_mask(self, src):
        """
        mask for masked attention
        :param src: seq_len, batch_size, embedding_dim
        :return: seq_len, seq_len
        """
        seq_len = src.size(1)
        return torch.triu(torch.ones(
            (seq_len, seq_len)), diagonal=1) == 1  # True-mask, False-unmask

    def _padding_mask(self, seq, pad_idx=0):
        """
        mask for padding
        :param seq: seq_len, batch_size, 1
        :param pad_idx: scalar
        :return: batch_size, seq_len
        """
        masked = (seq == pad_idx)
        return masked  # True-mask, False-unmask

    def forward(self, ):
        pass


class Encoder(TransModel):

    def __init__(self,
                 embed_layer,
                 temp_enc_layer,
                 embed_size=128,
                 num_layers=2,
                 dropout=0):
        super(Encoder, self).__init__()
        # self.embedding = embedding
        self.tms_enc = temp_enc_layer
        self.nheads = 2
        self.encoder_layer = nn.TransformerEncoderLayer(embed_size,
                                                        self.nheads,
                                                        batch_first=True,
                                                        dropout=dropout)
        self.rnn = nn.TransformerEncoder(self.encoder_layer,
                                         num_layers=num_layers)
        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)

    def forward(self, input_seq, x_tms_prob, if_tms_enc=False):
        device = input_seq.device
        mask_pm = self._padding_mask(input_seq).to(device)
        # mask_pm = (x_prob < 0.5) | mask_pm
        # memory = self.tms_enc(tms_encoding) + self.embed_layer(input_seq)
        memory = self.tms_enc(self.embed_layer(input_seq))
        # memory = self.embed_layer(input_seq)
        memory = self.rnn(memory, src_key_padding_mask=mask_pm)
        return memory, mask_pm, self._padding_mask(input_seq).to(device)


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, multiply_attn: Optional[Tensor] = None,) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.

          If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, multiply_attn=multiply_attn)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, multiply_attn=multiply_attn)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first']

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 layer_norm_eps=1e-5,
                 batch_first=False,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model,
                                            nhead,
                                            dropout=dropout,
                                            batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model,
                                                 nhead,
                                                 dropout=dropout,
                                                 batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                multiply_attn: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt,
                              tgt,
                              tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, weights = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            multiply_attn=multiply_attn)
        weights.size()
        # writer.add_image('mem_mask/1', memory_mask[0][None])
        # writer.add_image('mem_mask/2', memory_mask[1][None])
        # writer.add_image('mem_mask/3', memory_mask[2][None])
        # writer.add_image('weights/1', weights[0][None])
        # writer.add_image('weights/2', weights[1][None])
        # writer.add_image('weights/3', weights[2][None])
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                multiply_attn: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output,
                         memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         multiply_attn=multiply_attn)

        if self.norm is not None:
            output = self.norm(output)

        return output


class Decoder(TransModel):

    def __init__(self,
                 embed_layer,
                 temp_enc_layer,
                 embed_size,
                 num_layer,
                 nodeid2token,
                 dropout=0):
        super(Decoder, self).__init__()
        self.nhead = 2
        self.tms_enc = temp_enc_layer
        self.decoder_layer = TransformerDecoderLayer(d_model=embed_size,
                                                     nhead=self.nhead,
                                                     dropout=dropout,
                                                     batch_first=True)
        self.rnn = TransformerDecoder(self.decoder_layer, num_layers=num_layer)
        self.out = nn.Linear(d_model, nodeid2token.vocab_size)

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)

    def forward(self, input_step, memory, mask_pm, multiply_attn=None):
        device = input_step.device
        mask_s = self._sequence_mask(input_step).to(device)
        mask_p = self._padding_mask(input_step).to(device)
        # embedded = self.tms_enc(self.embed_layer(input_step))
        embedded = self.embed_layer(input_step)
        output = self.rnn(embedded,
                          memory,
                          tgt_mask=mask_s,
                          tgt_key_padding_mask=mask_p,
                          memory_key_padding_mask=mask_pm,
                          multiply_attn=multiply_attn)
        output = self.out(output)
        return output


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, denoiser, nodeid2token):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.denoiser = denoiser
        self.nodeid2token = nodeid2token
        self.output_size = nodeid2token.vocab_size
        self._SOS_token = nodeid2token.encode(['<s>']).item()

    def forward(self, low_high_seq, tms_recd, labels):
        # encoder
        input_seq = low_high_seq[:, -1]
        x_tms_low = tms_recd[:, 0]
        memory, mem_pad_mask, _ = self.encoder(input_seq, x_tms_low)
        out, inp = self.denoiser.denoise_score(low_high_seq, tms_recd)
        multiply_attn = self.denoiser.paded_seq(torch.sigmoid(out), inp).squeeze(2)
        multiply_attn = multiply_attn.repeat_interleave(3, dim=1)[:, 1: -1]
        multiply_attn = F.pad(multiply_attn, (0, input_seq.size(1) - multiply_attn.size(1)), "constant", 0)
        multiply_attn = multiply_attn[:, None, :].repeat(1, labels.size(1), 1)\
            .repeat_interleave(self.decoder.nhead, 0)
        # decoder
        logits = self.decoder(labels, memory, mem_pad_mask, multiply_attn)
        return logits, out.squeeze()

    @torch.no_grad()
    def infer_forwards(self, low_high_seq, tms_recd, label_length=40):
        input_seq = low_high_seq[:, -1]
        batch_size = input_seq.size(0)
        x_tms_low = tms_recd[:, 0]
        memory, mem_pad_mask, _ = self.encoder(input_seq, x_tms_low)
        out, inp = self.denoiser.denoise_score(low_high_seq, tms_recd)
        multiply_attn = self.denoiser.paded_seq(torch.sigmoid(out), inp).squeeze(2)
        multiply_attn = multiply_attn.repeat_interleave(3, dim=1)[:, 1: -1]
        multiply_attn = F.pad(multiply_attn, (0, input_seq.size(1) - multiply_attn.size(1)), "constant", 0)
        multiply_attn = multiply_attn[:, None, :].repeat(1, label_length, 1)\
            .repeat_interleave(self.decoder.nhead, 0)
        multiply_attn.requires_grad = False
        samples = torch.zeros([batch_size, label_length]).long().to(input_seq.device)
        samples[:, 0] = self._SOS_token
        for step in range(1, label_length):
            decoder_output = self.decoder(samples,
                                          memory,
                                          mem_pad_mask,
                                          multiply_attn=multiply_attn)
            pred = decoder_output.argmax(2)[:, step - 1]
            samples[:, step] = pred
        return samples


class GCNDeno(nn.Module):

    def __init__(self, embed, rec_embed, embed_size=128):
        super(GCNDeno, self).__init__()
        self.embed_layer = embed
        self.rec_embed = rec_embed
        self.encoder_layer = nn.ModuleList([
            ChebConv(embed_size, embed_size, 3),
            ChebConv(embed_size, embed_size, 3)
        ])
        self.bilinear = nn.Bilinear(embed_size + 512, embed_size + 512, 1)

    def forward(self, input_seq, recd_token):
        # encoder
        input = self.gen_graph_data(input_seq, recd_token)
        x = self.embed_layer(input.x)
        for gnn in self.encoder_layer:
            x = gnn(x, edge_index=input.edge_index)
            x = F.relu(x)
        out = torch.cat([x, self.rec_embed(input.app)], dim=1)
        return out, input
    
    def denoise_score(self, low_high_seq, tms_recd):
        input_seq_low, recd_token_low, input_seq_high, recd_token_high = low_high_seq[:, 0], tms_recd[:, 2], \
            low_high_seq[:, 1], tms_recd[:, 3]
        out, input = self.forward(input_seq_high, recd_token_high) 
        high_out = global_mean_pool(out, input.batch)
        out, input = self.forward(input_seq_low, recd_token_low)  
        return self.bilinear(out, high_out.index_select(0, input.batch)), input

    def paded_seq(self, out, input):
        return pad_sequence(unbatch(out, input.batch),
                            batch_first=True,
                            padding_value=0)

    def gen_graph_data(self, input_seq, recd_token):

        def construct_adj(item, recd):
            app_feat = F.normalize(self.rec_embed(recd[item > 0]), dim=1)
            A_app = app_feat @ app_feat.t()
            mean = torch.triu(A_app, 1)[torch.triu(A_app, 1) != 0].mean()
            A_app[A_app < mean] = 0
            return A_app

        def gen_data(item, recd):
            item = item[:recd.size(0)]
            adj = construct_adj(item.detach(), recd.detach())
            edge_index = adj.nonzero().t().contiguous()
            edge_weight = adj[edge_index[0], edge_index[1]]
            return Data(x=item[item > 0], app=recd[item > 0], 
                        edge_index=edge_index,
                        edge_weight=edge_weight)

        data = [gen_data(item, recd) for item, recd in zip(input_seq, recd_token)]
        return Batch.from_data_list(data)


if __name__ == '__main__':
    import torch
    import math
    import pickle
    import numpy as np
    import torch.nn as nn
    import torch.optim as optim
    from collections import Counter, defaultdict
    from tqdm import tqdm
    # from dataloader_realtest import data_loader
    from dataloader import data_loader
    from torch.utils.tensorboard import SummaryWriter
    import pandas as pd
    import yaml

    config_filename = f'../models/cfg.yaml'
    configs = yaml.load(open(config_filename), Loader=yaml.FullLoader)
    filename = 'real_test_tfm_tklet_deno_3d'
    writer = SummaryWriter(f'tblogs/{filename}')
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    d_model, num_layer = 128, 2
    train_loader, test_loader, nodeid2token, recd_emb = data_loader(128, configs)
    print(len(train_loader), len(test_loader))
    embed = NodeEmbeding(nodeid2token).to(device)
    temp_enc = PositionEncoding(d_model).to(device)
    encoder = Encoder(embed,
                      temp_enc,
                      embed_size=d_model,
                      num_layers=num_layer,
                      dropout=0.1).to(device)
    decoder = Decoder(embed,
                      temp_enc,
                      d_model,
                      num_layer,
                      nodeid2token,
                      dropout=0.1).to(device)
    denoiser = GCNDeno(embed, recd_emb, embed_size=128).to(device)
    pre_model = Seq2Seq(encoder, decoder, denoiser, nodeid2token=nodeid2token).to(device)
    pre_model.encoder.load_state_dict(torch.load(configs['save_path'] + 'pretrain_model_200.pth')['enc_state_dict'])
    pre_model.decoder.load_state_dict(torch.load(configs['save_path'] + 'pretrain_model_200.pth')['dec_state_dict'])
    optimizer = torch.optim.AdamW(pre_model.parameters(), lr=1e-3)

    loss_func = nn.CrossEntropyLoss(ignore_index=0).to(device)
    deno_loss = nn.BCEWithLogitsLoss().to(device)
    num_epochs = configs['epochs']
    for epoch in tqdm(range(num_epochs)):
        with torch.no_grad():
            loss_e = 0
            pre_model.eval()
            preds = []
            cam_inp = []
            cam_tms = []
            gt_cam = []
            opod_acc = 0
            deno_acc = 0
            for idx, (x_rcds, x_prob, y_traj) in enumerate(test_loader):
                x_rcds, x_prob, y_traj = x_rcds.to(device), x_prob.to(device), y_traj.to(device)
                logits, out = pre_model(x_rcds, x_prob, y_traj[:, :-1])
                denolabel = x_prob[:, -1]
                loss_deno = deno_loss(out, denolabel[denolabel!=-1].float())
                acc = ((torch.sigmoid(out) > 0.5).float() == denolabel[denolabel!=-1].float()).sum() / out.size(0)
                opod_acc += (torch.ones_like(torch.sigmoid(out)).float() == denolabel[denolabel!=-1].float()).sum() / out.size(0)
                deno_acc += acc
                if (epoch == num_epochs - 1) and idx == 0:
                    batch_size = x_rcds.size(0)
                    pads, eos, bos = nodeid2token.encode(
                        ['<pad>', '<\s>', '<s>']).to(device)
                    Y = torch.ones(batch_size, 1).to(device).long() * bos
                    pred = pre_model.infer_forwards(x_rcds, x_prob)
                    preds.append(pred.cpu().numpy())
                    cam_inp.append(x_rcds.cpu().numpy()[:, -1])
                    cam_tms.append(x_prob.cpu().numpy())
                    gt_cam.append(y_traj.cpu().numpy())
                loss = loss_func(logits.reshape(-1, logits.size(-1)), y_traj[:, 1:].reshape(-1, ))
                loss += loss_deno
                loss_e += loss.item()
            writer.add_scalar('loss/Test', loss_e, epoch)
            writer.add_scalar('loss/opod_acc', (opod_acc / (idx+1)).item(), epoch)
            writer.add_scalar('loss/denoise_acc', (deno_acc / (idx+1)).item(), epoch)
        print(loss_e)
        pre_model.train()
        loss_e = 0
        for idx, (x_rcds, x_prob, y_traj) in enumerate(train_loader):
            x_rcds, x_prob, y_traj = x_rcds.to(device), x_prob.to(device), y_traj.to(device)
            logits, out = pre_model(x_rcds, x_prob, y_traj[:, :-1])
            loss = loss_func(logits.reshape(-1, logits.size(-1)),
                             y_traj[:, 1:].reshape(-1, ))
            denolabel = x_prob[:, -1]
            loss_deno = deno_loss(out, denolabel[denolabel!=-1].float())
            loss += loss_deno
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_e += loss.item()
        writer.add_scalar('loss/Train', loss_e, epoch)

    save_path = '../results/'
    pickle.dump({'pred': np.concatenate(preds), 'cam_tms': np.concatenate(cam_tms), 'cam_id': np.concatenate(cam_inp), 'grth': np.concatenate(gt_cam)},\
        open(save_path + f"{filename}.pkl", 'wb'))
    writer.close()