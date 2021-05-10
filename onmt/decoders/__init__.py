"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder

str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder}

__all__ = ["DecoderBase", "StdRNNDecoder",
           "InputFeedRNNDecoder", "str2dec"]
