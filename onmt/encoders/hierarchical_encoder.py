import torch
import torch.nn as nn

from onmt.encoders.rgcn_gate_encoder import RGCNGateEncoder
from onmt.encoders.rnn_encoder import RNNEncoder


class HierarchicalEncoder3(nn.Module):

    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size, dropout, embeddings, use_bridge,
                 gnn_type, gnn_layers, speaker_type):
        super().__init__()

        self.embeddings = embeddings

        self.rnn = RNNEncoder(rnn_type, bidirectional, num_layers, hidden_size, speaker_type, dropout,
                              self.embeddings, use_bridge)

        self.gnn = RGCNGateEncoder(hidden_size)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.bridge,
            opt.gnn_type,
            opt.gnn_layers,
            opt.speaker_type)

    def forward(self, src, seg, speaker, adj_coo, edge_types, rels, lengths=None):
        """

        :param src: torch.LongTensor (meeting_words_seq_len, batch, features)
        :param seg: List (batch,?)
        :param speaker: List (batch,?)
        :param adj_coo:
        :param rels:
        :param lengths:
        :return:
        """

        """
        Transform src format
        """
        assert src.size(2) == 1
        src = src.squeeze(-1).transpose(0, 1).contiguous()  # (batch_size, meeting_words_seq_len)
        batch_size = src.size(0)
        meeting_words_seq_len = src.size(1)  # max len of all words in the batch

        batch_utterances_list = []
        batch_utterances_lens = []
        batch_meeting_lens = []
        batch_speaker_list = []
        for index, meeting in enumerate(src):  # for one meeting
            batch_meeting_lens.append(len(seg[index]) - 1)  # num of utterance, 0 2 5 represent two utterance.
            for i in range(len(seg[index]) - 1):
                batch_utterances_list.append(meeting[seg[index][i]:seg[index][i + 1]])
                batch_speaker_list.append(speaker[index][seg[index][i]:seg[index][i + 1]])
                batch_utterances_lens.append(len(batch_utterances_list[-1]))
                assert batch_utterances_lens[-1] == seg[index][i + 1] - seg[index][
                    i], "utterance len is not same as seg"

        batch_utterances_lens = torch.tensor(
            batch_utterances_lens).cuda()  # len of utterance of evert utterance in one batch

        speakers = torch.zeros(len(batch_utterances_lens), max(batch_utterances_lens), dtype=torch.long).cuda()
        for index, s in enumerate(batch_speaker_list):
            speakers[index, :batch_utterances_lens[index]] = torch.tensor(s)

        utterances = torch.ones(len(batch_utterances_lens), max(batch_utterances_lens), dtype=torch.long).cuda()
        for index, utterance in enumerate(batch_utterances_list):
            utterances[index, :batch_utterances_lens[index]] = utterance

        lens_new_order, idxs = batch_utterances_lens.sort(descending=True)
        utterances_new_order = utterances.index_select(0, idxs)  # sorted by length
        speakers_new_order = speakers.index_select(0, idxs)

        """
        Utterance-level RNN
        """
        temp_enc_state, temp_memory_bank, temp_lengths = self.rnn(utterances_new_order.transpose(0, 1).unsqueeze(-1),
                                                                  speakers_new_order.transpose(0, 1), lens_new_order)

        """
        Back order hidden state and memory bank
        """
        _, reverse_idxs = idxs.sort()

        utterances_enc_hidden_state = temp_enc_state[0].index_select(1, reverse_idxs)
        words_memory_bank = temp_memory_bank.index_select(1, reverse_idxs)

        """
        Prepare for graph neural networks
        """
        utterances_enc_hidden_state = utterances_enc_hidden_state.squeeze(0)  # (utterance num, hidden_size)
        meeting_utterance_enc_hidden_state = utterances_enc_hidden_state.split(batch_meeting_lens)

        """
        Graph neural networks
        """
        graph_res_list = self.gnn(meeting_utterance_enc_hidden_state, adj_coo, edge_types, rels, batch_meeting_lens)

        """
        Create word-level output 
        """
        words_memory_bank = words_memory_bank.transpose(0,
                                                        1).contiguous()  # (num utterances, max_len, hidden_size)
        words_memory_bank_list = words_memory_bank.split(batch_meeting_lens)

        utterances_lens_seg = batch_utterances_lens.split(batch_meeting_lens)
        memory_bank = torch.zeros((batch_size, meeting_words_seq_len, words_memory_bank.size(-1))).cuda()

        for i, w in enumerate(words_memory_bank_list):
            item_res = []
            for j, utterance in enumerate(w):  # (1, max_utterance_len, hidden_size)
                utterance = utterance.squeeze(0)  # (max_utterance_len, hidden_size)
                utterance = utterance[:utterances_lens_seg[i][j]]  # (true_len, hidden_size)
                item_res.append(utterance)
            item_final_res = torch.cat(item_res, 0)  # (len, hidden_size)
            assert item_final_res.size(0) == lengths[i]
            memory_bank[i, :item_final_res.size(0), :] = item_final_res  # (batch, meeting_words_seq_len, hidden_size)

        memory_bank = memory_bank.transpose(0, 1).contiguous()  # (meeting_words_seq_len, batch, hidden_size)

        """
        Create utterance-level representation
        """
        update_utterance_res = []
        global_reps = []
        for index, g in enumerate(graph_res_list):
            global_rep = g[-1]  # index of global node
            global_reps.append(global_rep)
            g = g[:batch_meeting_lens[index]]  # (num_utterances, hidden_size)

            update_utterance_res.append(g)

        memory_bank_utterance = torch.zeros(
            (batch_size, meeting_words_seq_len, update_utterance_res[0].size(-1))).cuda()
        for i, u in enumerate(update_utterance_res):  # (num_utterances, hidden_size)
            previous_count = 0
            for j, one in enumerate(u):  # (1,hidden_size)
                one = one.repeat(utterances_lens_seg[i][j], 1)  # word_num, hidden_size
                memory_bank_utterance[i, previous_count:utterances_lens_seg[i][j] + previous_count] = one
                previous_count = previous_count + utterances_lens_seg[i][j]
        memory_bank_utterance = memory_bank_utterance.transpose(0, 1).contiguous()

        # use utterances_lens_seg to create a matrix used for hier attn
        hier_matrix = torch.zeros((batch_size, meeting_words_seq_len)).cuda()
        for i, one in enumerate(utterances_lens_seg):
            previous_count = 0
            for j, item in enumerate(one):
                hier_matrix[i, previous_count:utterances_lens_seg[i][j] + previous_count] = item
                previous_count = previous_count + utterances_lens_seg[i][j]

        encoder_hidden_final = torch.stack(global_reps, 0)  # (batch, hidden_size)
        encoder_cell_final = encoder_hidden_final.clone()
        encoder_hidden_final = encoder_hidden_final.unsqueeze(0)
        encoder_cell_final = encoder_cell_final.unsqueeze(0)

        return (encoder_hidden_final,
                encoder_cell_final), memory_bank, lengths, memory_bank_utterance, lengths, hier_matrix
