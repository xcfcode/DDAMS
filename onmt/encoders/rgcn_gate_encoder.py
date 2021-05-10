
import torch
import torch.nn as nn
from onmt.const import relations_vocab
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from onmt.encoders.rgcn_gate_layer import RGCNConvWithGate


class RGCNGateEncoder(torch.nn.Module):
    def __init__(self, hidden_size):
        super(RGCNGateEncoder, self).__init__()
        self.relation_embeddings = nn.Embedding(len(relations_vocab), hidden_size)
        self.relation_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.conv1 = RGCNConvWithGate(hidden_size, hidden_size, 6, num_bases=30)
        self.conv2 = RGCNConvWithGate(hidden_size, hidden_size, 6, num_bases=30)

    def forward(self, meeting_utterance_enc_hidden_states, adj_coos, edge_types, rels, meeting_lens):
        # create graph batch
        list_geometric_data = []
        seg_len = []
        for meeting_utterance_enc_hidden_state, adj_coo, edge_type, rel in zip(meeting_utterance_enc_hidden_states,
                                                                               adj_coos, edge_types,
                                                                               rels):
            rel = torch.LongTensor(rel).cuda()
            rel_embed = self.relation_embeddings(rel)  # (relation num, hidden size)
            emb = torch.cat((meeting_utterance_enc_hidden_state, rel_embed),
                            0)  # (utterance num + relation num, hidden_size)

            assert emb.size(0) == max(adj_coo[0]) + 1, "make sure the emb == adj matrix"

            seg_len.append(emb.size(0))

            edge_index = torch.tensor(adj_coo, dtype=torch.long).cuda()
            edge_type = torch.tensor(edge_type, dtype=torch.long).cuda()

            data = Data(x=emb, edge_index=edge_index)
            data.edge_type = edge_type  # edge_type
            list_geometric_data.append(data)

        batch_geometric = Batch.from_data_list(list_geometric_data).to('cuda')

        x = F.relu(self.conv1(batch_geometric.x, batch_geometric.edge_index, batch_geometric.edge_type))
        x = self.conv2(x, batch_geometric.edge_index, batch_geometric.edge_type)
        x = x.split(seg_len)


        return x
