from torch import Tensor
import copy
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList

from torch.nn.modules.transformer import MultiheadAttention, _get_activation_fn




class TransEncoderCrossAtt(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransEncoderCrossAtt, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm




    def forward(self, query: Tensor,
                key_value: Tensor,
                att_mask = None,
                key_padding_mask = None) -> Tensor:


        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = query

        for mod in self.layers:
            output = mod(query=output, key_value=key_value, key_padding_mask = key_padding_mask)
            # output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output



class TransWithCrossAtt(nn.Module):
    def __init__(self, args, dropout = 0.1,
                 activation = F.relu,
                 layer_norm_eps=1e-5,
                 norm_first=False,):
        super(TransWithCrossAtt, self).__init__()
        self.cross_att = MultiheadAttention(embed_dim=args.att_hidden_size,
                                            num_heads=args.num_head, batch_first=True, dropout=dropout)

        self.linear1 = nn.Linear(args.att_hidden_size, args.dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(args.dim_feedforward, args.att_hidden_size)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(args.att_hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(args.att_hidden_size, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, query: Tensor,
                key_value: Tensor,
                att_mask = None,
                key_padding_mask = None):


        x = query
        if self.norm_first:
            x = x + self._crossatt_block(self.norm1(x), key_value, att_mask, key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._crossatt_block(self.norm1(x), key_value, att_mask, key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # cross-attention block
    def _crossatt_block(self, x: Tensor,
                        key_value: Tensor,
                        attn_mask,
                        key_padding_mask):
        x = self.cross_att(x, key_value, key_value,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])