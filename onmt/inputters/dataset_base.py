# coding: utf-8

from itertools import chain, starmap
from collections import Counter

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example
from torchtext.vocab import Vocab

from onmt.const import relations_vocab

"""
def _join_dicts : 多个dict合并为一个dict
def _dynamic_dict : 为pointer做准备，每一个example有一个自己的src_map、src_ex_vocab和alignment
class Dataset : torchtext的dataset类，包含了example、fields
"""


def _join_dicts(*args):
    """

    Args:
        dictionaries with disjoint keys.

    Returns:
        a single dictionary that has the union of these keys.
    """

    return dict(chain(*[d.items() for d in args]))


def _dynamic_dict(example, src_field, tgt_field):
    """Create copy-vocab and numericalize with it.



    In-place adds ``"src_map"`` to ``example``. That is the copy-vocab
    numericalization of the tokenized ``example["src"]``. If ``example``
    has a ``"tgt"`` key, adds ``"alignment"`` to example. That is the
    copy-vocab numericalization of the tokenized ``example["tgt"]``. The
    alignment has an initial and final UNK token to match the BOS and EOS
    tokens.

    Args:
        example (dict): An example dictionary with a ``"src"`` key and
            maybe a ``"tgt"`` key. (This argument changes in place!)
        src_field (torchtext.data.Field): Field object.
        tgt_field (torchtext.data.Field): Field object.

    Returns:
        torchtext.data.Vocab and ``example``, changed as described.
    """

    src = src_field.tokenize(example["src"])
    # make a small vocab containing just the tokens in the source sequence
    unk = src_field.unk_token
    pad = src_field.pad_token
    src_ex_vocab = Vocab(Counter(src), specials=[unk, pad])
    unk_idx = src_ex_vocab.stoi[unk]
    # Map source tokens to indices in the dynamic dict.
    src_map = torch.LongTensor([src_ex_vocab.stoi[w] for w in src])
    example["src_map"] = src_map
    example["src_ex_vocab"] = src_ex_vocab

    if "tgt" in example:
        tgt = tgt_field.tokenize(example["tgt"])
        mask = torch.LongTensor(
            [unk_idx] + [src_ex_vocab.stoi[w] for w in tgt] + [unk_idx])
        example["alignment"] = mask
    return src_ex_vocab, example


def make_graph(relation_str, utterances_num, relations_vocab):
    """

    :param relation_str:
    :param utterances_num:
    :param relations_vocab:
    :return:
    """
    rel_list = relation_str.split("\t")
    rel_list = [[int(rel.strip().split()[0]), rel.strip().split()[1], int(rel.strip().split()[2])] for rel in
                rel_list]

    rels = []
    adj_size = utterances_num + 1 + len(rel_list)  # utterance数量，global节点，边节点

    """make graph"""
    edge_list = []
    utterance_egde_list = []

    """total graph"""
    for i in range(adj_size):
        edge_list.append([adj_size - 1, i, 0])

    # Self
    for i in range(adj_size):
        edge_list.append([i, i, 1])

    """discourse-aware r-gcn"""
    for rel in rel_list:
        rels.append(relations_vocab[rel[1].strip()])
        a = rel[0]  # head
        b = rel[2]  # tail
        c = utterances_num + len(rels) - 1  # idx of this REL

        # default
        edge_list.append([a, c, 2])  # in-discourse type
        edge_list.append([c, b, 3])  # out-discourse type

        # reverse
        edge_list.append([b, c, 4])  # reverse in-discourse type
        edge_list.append([c, a, 5])  # reverse out-discourse type


    rels.append(relations_vocab["global"])
    edge_list = sorted(edge_list, key=lambda x: (x[0], x[1], x[2]))
    edge = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index, edge_type = edge[:2], edge[2]
    adj_coo = edge_index.tolist()
    edge_type = edge_type.tolist()

    return adj_coo, edge_type, rels


class Dataset(TorchtextDataset):
    """Contain data and process it.

    A dataset is an object that accepts sequences of raw data (sentence pairs
    in the case of machine translation) and fields which describe how this
    raw data should be processed to produce tensors. When a dataset is
    instantiated, it applies the fields' preprocessing pipeline (but not
    the bit that numericalizes it or turns it into batch tensors) to the raw
    data, producing a list of :class:`torchtext.data.Example` objects.
    torchtext's iterators then know how to use these examples to make batches.

    Args:
        fields (dict[str, Field]): a dict with the structure
            returned by :func:`onmt.inputters.get_fields()`. Usually
            that means the dataset side, ``"src"`` or ``"tgt"``. Keys match
            the keys of items yielded by the ``readers``, while values
            are lists of (name, Field) pairs. An attribute with this
            name will be created for each :class:`torchtext.data.Example`
            object and its value will be the result of applying the Field
            to the data that matches the key. The advantage of having
            sequences of fields for each piece of raw input is that it allows
            the dataset to store multiple "views" of each input, which allows
            for easy implementation of token-level features, mixed word-
            and character-level models, and so on. (See also
            :class:`onmt.inputters.TextMultiField`.)
        readers (Iterable[onmt.inputters.DataReaderBase]): Reader objects
            for disk-to-dict. The yielded dicts are then processed
            according to ``fields``.
        data (Iterable[Tuple[str, Any]]): (name, ``data_arg``) pairs
            where ``data_arg`` is passed to the ``read()`` method of the
            reader in ``readers`` at that position. (See the reader object for
            details on the ``Any`` type.)
        dirs (Iterable[str or NoneType]): A list of directories where
            data is contained. See the reader object for more details.
        sort_key (Callable[[torchtext.data.Example], Any]): A function
            for determining the value on which data is sorted (i.e. length).
        filter_pred (Callable[[torchtext.data.Example], bool]): A function
            that accepts Example objects and returns a boolean value
            indicating whether to include that example in the dataset.

    Attributes:
        src_vocabs (List[torchtext.data.Vocab]): Used with dynamic dict/copy
            attention. There is a very short vocab for each src example.
            It contains just the source words, e.g. so that the generator can
            predict to copy them.
    """

    def __init__(self, fields, readers, data, dirs, sort_key,
                 filter_pred=None):
        self.sort_key = sort_key
        can_copy = 'src_map' in fields and 'alignment' in fields

        read_iters = [r.read(dat[1], dat[0], dir_) for r, dat, dir_
                      in zip(readers, data, dirs)]

        # self.src_vocabs is used in collapse_copy_scores and Translator.py
        self.src_vocabs = []
        examples = []
        for ex_dict in starmap(_join_dicts, zip(*read_iters)):
            assert len(ex_dict["src"].strip().split()) == len(ex_dict["speaker"].strip().split())
            assert len(ex_dict["src"].strip().split()) == int(ex_dict["seg"].strip().split()[-1])

            if can_copy:
                src_field = fields['src']
                tgt_field = fields['tgt']
                # this assumes src_field and tgt_field are both text
                src_ex_vocab, ex_dict = _dynamic_dict(
                    ex_dict, src_field.base_field, tgt_field.base_field)
                self.src_vocabs.append(src_ex_vocab)

            """
            Prepare for graph neural networks(GNN)
            """
            relation_str = ex_dict["relation"]
            utterances_num = len(ex_dict["seg"].split()) - 1
            adj_coo, edge_type, rels = make_graph(relation_str, utterances_num, relations_vocab)
            ex_dict["adj_coo"] = adj_coo
            ex_dict["rels"] = rels
            ex_dict["edge_type"] = edge_type

            """
            String to List
            """
            ex_dict["seg"] = [int(item) for item in ex_dict["seg"].strip().split()]
            ex_dict["speaker"] = [int(item) for item in ex_dict["speaker"].strip().split()]

            ex_fields = {k: [(k, v)] for k, v in fields.items() if
                         k in ex_dict}
            ex = Example.fromdict(ex_dict, ex_fields)
            examples.append(ex)

        # fields needs to have only keys that examples have as attrs
        fields = []
        for _, nf_list in ex_fields.items():
            assert len(nf_list) == 1
            fields.append(nf_list[0])
        super(Dataset, self).__init__(examples, fields, filter_pred)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)
