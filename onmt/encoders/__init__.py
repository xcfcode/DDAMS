"""Module defining encoders."""
from onmt.encoders.hierarchical_encoder import HierarchicalEncoder3
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.rnn_encoder import RNNEncoder

str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "hier3": HierarchicalEncoder3}

__all__ = ["EncoderBase", "RNNEncoder", "str2enc"]
