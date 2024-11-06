import torch
import torch.nn as nn
import torch_geometric
from gvp_transformer_pdbbind.gvp import GVP, GVPConvLayer, LayerNorm
from gvp_transformer_pdbbind.ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
from .RetNet import RetNet
from mamba_ssm import Mamba2
import numpy as np

# conv_filters = [[1, 32], [3, 32], [5, 64], [7, 128]]
# embedding_size = output_dim = 512
# d_ff = 256
# n_heads = 8
# d_k = 16
# n_layer = 1

class ProtGVPModel(nn.Module):
    def __init__(self,
                 node_in_dim=[6, 3], node_h_dim=[128, 64],
                 edge_in_dim=[39, 1], edge_h_dim=[32, 1],
                 num_layers=3, drop_rate=0.1
                 ):
        """
        Parameters
        ----------
        node_in_dim : list of int
            Input dimension of drug node features (si, vi).
            Scalar node feartures have shape (N, si).
            Vector node features have shape (N, vi, 3).
        node_h_dims : list of int
            Hidden dimension of drug node features (so, vo).
            Scalar node feartures have shape (N, so).
            Vector node features have shape (N, vo, 3).
        """
        super(ProtGVPModel, self).__init__()
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def pyg_split(self, batched_data, feats):
        # print(batched_data)
        # print(batched_data.batch)
        device = feats.device
        # ptr 属性定义了每个图的边索引在批次中的起始位置，图的个数等于 ptr 张量的长度减1
        batch_size = batched_data.ptr.size(0) - 1
        # 获取 batch 属性，这个属性记录了每个节点所属图的索引
        node_to_graph_idx = batched_data.batch
        num_nodes_per_graph = torch.bincount(node_to_graph_idx)
        max_num_nodes = int(num_nodes_per_graph.max())

        batch = torch.cat(
            [torch.full((1, x.type(torch.int)), y) for x, y in zip(num_nodes_per_graph, range(batch_size))],
            dim=1).reshape(-1).type(torch.long).to(device)
        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes_per_graph.cumsum(dim=0)])
        idx = torch.arange(len(node_to_graph_idx), dtype=torch.long, device=device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        size = [batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats
        out = out.view([batch_size, max_num_nodes] + list(feats.size())[1:])
        return out

    def forward(self, xp):
        # Unpack input data
        h_V = (xp.node_s, xp.node_v)
        # print(xp.node_s.shape, xp.node_v.shape, xp.edge_s.shape, xp.edge_v.shape)
        # print(xp)
        h_E = (xp.edge_s, xp.edge_v)
        edge_index = xp.edge_index
        protein_seq = xp.seq
        batch = xp.batch

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
        out = self.pyg_split(xp, out)
        # out = torch_geometric.nn.global_add_pool(out, batch)

        return out


class DrugGVPModel(nn.Module):
    def __init__(self,
                 node_in_dim=[66, 1], node_h_dim=[128, 128],
                 edge_in_dim=[16, 1], edge_h_dim=[32, 1],
                 num_layers=None, drop_rate=0.1
                 ):
        """
        Parameters
        ----------
        node_in_dim : list of int
            Input dimension of drug node features (si, vi).
            Scalar node feartures have shape (N, si).
            Vector node features have shape (N, vi, 3).
        node_h_dim : list of int
            Hidden dimension of drug node features (so, vo).
            Scalar node feartures have shape (N, so).
            Vector node features have shape (N, vo, 3).
        edge_in_dim : list of int
            Input dimension of drug node features (si, vi).
            Scalar node feartures have shape (N, si).
            Vector node features have shape (N, vi, 3).
        edge_h_dim : list of int
            Hidden dimension of drug node features (so, vo).
            Scalar node feartures have shape (N, so).
            Vector node features have shape (N, vo, 3).
        """
        super(DrugGVPModel, self).__init__()
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def pyg_split(self, batched_data, feats):
        # print(batched_data)
        # print(batched_data.batch)
        device = feats.device
        # ptr 属性定义了每个图的边索引在批次中的起始位置，图的个数等于 ptr 张量的长度减1
        batch_size = batched_data.ptr.size(0) - 1
        # 获取 batch 属性，这个属性记录了每个节点所属图的索引
        node_to_graph_idx = batched_data.batch
        num_nodes_per_graph = torch.bincount(node_to_graph_idx)
        max_num_nodes = int(num_nodes_per_graph.max())

        batch = torch.cat(
            [torch.full((1, x.type(torch.int)), y) for x, y in zip(num_nodes_per_graph, range(batch_size))],
            dim=1).reshape(-1).type(torch.long).to(device)
        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes_per_graph.cumsum(dim=0)])
        idx = torch.arange(len(node_to_graph_idx), dtype=torch.long, device=device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        size = [batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats
        out = out.view([batch_size, max_num_nodes] + list(feats.size())[1:])
        return out

    def forward(self, xd):
        # Unpack input data
        h_V = (xd.node_s, xd.node_v)
        # print(xp.node_s.shape, xp.node_v.shape, xp.edge_s.shape, xp.edge_v.shape)
        h_E = (xd.edge_s, xd.edge_v)
        edge_index = xd.edge_index
        batch = xd.batch

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)

        out = self.W_out(h_V)
        out = self.pyg_split(xd, out)
        return out


class BilinearPooling(nn.Module):
    def __init__(self, in_channels, out_channels, c_m, c_n):
        super().__init__()

        self.convA = nn.Conv1d(in_channels, c_m, kernel_size=1, stride=1, padding=0)
        self.convB = nn.Conv1d(in_channels, c_n, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(c_m, out_channels, bias=True)

    def forward(self, x):
        '''
        x: (batch, channels, seq_len)
        A: (batch, c_m, seq_len)
        B: (batch, c_n, seq_len)
        att_maps.permute(0, 2, 1): (batch, seq_len, c_n)
        global_descriptors: (batch, c_m, c_n)
        '''
        # print(x.shape)
        A = self.convA(x)
        # print(A.shape)
        B = self.convB(x)
        att_maps = F.softmax(B, dim=-1)
        global_descriptors = torch.bmm(A, att_maps.permute(0, 2, 1))
        global_descriptor = torch.mean(global_descriptors, dim=-1)
        out = self.linear(global_descriptor).unsqueeze(1)

        return out


class MutualAttentation(nn.Module):
    def __init__(self, in_channels, att_size, c_m, c_n):
        super().__init__()
        self.bipool = BilinearPooling(in_channels, in_channels, c_m, c_n)
        self.linearS = nn.Linear(in_channels, att_size)
        self.linearT = nn.Linear(in_channels, att_size)

    def forward(self, source, target):
        '''
        source: (batch, channels, seq_len)
        target: (batch, channels, seq_len)
        global_descriptor: (batch, 1, channels)
        '''
        global_descriptor = self.bipool(source)
        # print("global_descriptor: ", global_descriptor.shape)
        target_org = target
        target = self.linearT(target.permute(0, 2, 1)).permute(0, 2, 1)
        # print("target: ", target.shape)
        global_descriptor = self.linearS(global_descriptor)
        # print("global_descriptor: ", global_descriptor.shape)
        att_maps = torch.bmm(global_descriptor, target)
        # print("att_maps: ", att_maps.shape)
        att_maps = F.sigmoid(att_maps)
        out_target = torch.add(target_org, torch.mul(target_org, att_maps))
        out_target = F.relu(out_target)
        # print("out_target: ", out_target.shape)

        return out_target

class ConvEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, conv_filters, output_dim, type):
        super().__init__()
        if type == 'seq':
            self.embed = nn.Embedding(vocab_size, embedding_size)

        elif type == 'poc':
            self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        self.convolutions = nn.ModuleList()
        for kernel_size, out_channels in conv_filters:
            conv = nn.Conv1d(embedding_size, out_channels, kernel_size, padding = (kernel_size - 1) // 2)
            self.convolutions.append(conv)
        # The dimension of concatenated vectors obtained from multiple one-dimensional convolutions
        self.num_filters = sum([f[1] for f in conv_filters])
        self.projection = nn.Linear(self.num_filters, output_dim)


    def forward(self, inputs):
        # embeds = self.embed(inputs)
        embeds = self.embed(inputs).transpose(-1,-2) # (batch_size, embedding_size, seq_len)
        conv_hidden = []
        for layer in self.convolutions:
            conv = F.relu(layer(embeds))
            conv_hidden.append(conv)
        res_embed = torch.cat(conv_hidden, dim = 1).transpose(-1,-2) # (batch_size, seq_len, num_filters)
        embeds = self.projection(res_embed)
        return embeds


def get_attn_pad_mask(seq_q, seq_k):

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

conv_filters = [[1, 32], [3, 32], [5, 64], [7, 128]]
class Seq_Encoder(nn.Module):
    def __init__(self, embedding_dim=None):
        super().__init__()
        self.seq_emb = ConvEmbedding(vocab_size=22, embedding_size=embedding_dim,
                                     conv_filters=conv_filters, output_dim=embedding_dim, type='seq')
        self.poc_emb = ConvEmbedding(vocab_size=22, embedding_size=embedding_dim,
                                     conv_filters=conv_filters, output_dim=embedding_dim, type='poc')
        self.mamba2 = Mamba2(d_model=embedding_dim, d_state=64, d_conv=4, expand=2) # 256, 64 （128，64）, 4 (2～4), 2
        # self.retnet = RetNet(layers=1, hidden_dim=embedding_dim, ffn_size=embedding_dim // 2, heads=8, double_v_dim=False)
        self.linear = nn.Linear(embedding_dim * 2, embedding_dim // 2) # embedding_dim // 2 确保了除法操作的结果是一个整数
        # self.linear = nn.Linear(embedding_dim, embedding_dim // 2)  # embedding_dim // 2 确保了除法操作的结果是一个整数

    def forward(self, seq_input, poc_input):
        global_emb = self.seq_emb(seq_input) # (128, 1024, 256)
        # global_feats = self.retnet(F.relu(global_emb)) # (128, 1024, 256)
        global_feats = self.mamba2(F.relu(global_emb))  # (128, 1024, 256)
        local_emb = self.poc_emb(poc_input) # (128, 1024, 256)
        # local_feats = self.retnet(F.relu(local_emb)) # (128, 1024, 256)
        local_feats = self.mamba2(F.relu(local_emb))  # (128, 1024, 256)

        # output_emb = self.seq_emb(seq_input) + self.poc_emb(poc_input)  # (128, 1024, 256)
        # # local_feats = self.retnet(F.relu(local_emb)) # (128, 1024, 256)
        # output_emb = self.mamba2(F.relu(output_emb))  # (128, 1024, 256)

        output = torch.cat((global_feats, local_feats), dim=-1) # (128, 1024, 512)
        output = self.linear(output) # (128, 1024, 128)
        return output


class DTAModel(nn.Module):
    def __init__(self,
                 drug_node_in_dim=[66, 1], drug_node_h_dims=[128, 64],
                 drug_edge_in_dim=[16, 1], drug_edge_h_dims=[32, 1],
                 drug_fc_dims=[1024, 128],
                 prot_node_in_dim=[6, 3], prot_node_h_dims=[128, 64],
                 prot_edge_in_dim=[39, 1], prot_edge_h_dims=[32, 1],
                 prot_fc_dims=[1024, 128], bcn_h_dim=128,
                 mlp_dims=[1024, 512], mlp_dropout=0.25):
        super(DTAModel, self).__init__()

        self.drug_GVP = DrugGVPModel(
            node_in_dim=drug_node_in_dim, node_h_dim=drug_node_h_dims,
            edge_in_dim=drug_edge_in_dim, edge_h_dim=drug_edge_h_dims,
            num_layers=1
        )
        compound_dim = drug_node_h_dims[0]
        protein_dim = prot_node_h_dims[0]

        self.prot_GVP = ProtGVPModel(
            node_in_dim=prot_node_in_dim, node_h_dim=prot_node_h_dims,
            edge_in_dim=prot_edge_in_dim, edge_h_dim=prot_edge_h_dims,
            num_layers=3
        )
        # 128, 1024, 128
        self.drug_fc = self.get_fc_layers(
            [compound_dim] + drug_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)
        # 128, 1024, 128
        self.prot_fc = self.get_fc_layers(
            [protein_dim] + prot_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

        self.prot_mutual_attention = MutualAttentation(compound_dim, protein_dim, c_m=protein_dim, c_n=4)
        self.drug_mutual_attention = MutualAttentation(protein_dim, compound_dim, c_m=compound_dim, c_n=4)

        self.seq_encoder = Seq_Encoder(embedding_dim=256)  # 256


        # 128+128, 1024, 512, 1
        self.top_fc = self.get_fc_layers(
            [drug_fc_dims[-1] + prot_fc_dims[-1]] + mlp_dims + [1],
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

        self.bcn = weight_norm(
            BANLayer(v_dim=protein_dim, q_dim=compound_dim, h_dim=bcn_h_dim, head_num=6),
            name='h_mat', dim=None)

        # 128, 1024, 512, 1
        self.mlp = self.get_fc_layers(
            [bcn_h_dim * 3] + mlp_dims + [1],
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

    def get_fc_layers(self, hidden_sizes,
                      dropout=0, batchnorm=False,
                      no_last_dropout=True, no_last_activation=True):
        # act_fn = torch.nn.LeakyReLU()
        act_fn = torch.nn.ReLU()
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if not no_last_activation or i != len(hidden_sizes) - 2:
                layers.append(act_fn)
            if dropout > 0:
                if not no_last_dropout or i != len(hidden_sizes) - 2:
                    layers.append(nn.Dropout(dropout))
            if batchnorm and i != len(hidden_sizes) - 2:
                layers.append(nn.BatchNorm1d(out_dim))
        return nn.Sequential(*layers)


    def forward(self, xd, xp, protein_seq, pocket_seq):
        # compound_feats = self.drug_GVP(xd).permute(0, 2, 1) # [batch_size, hidden_dim, seq_len]
        # protein_feats = self.prot_GVP(xp).permute(0, 2, 1) # [batch_size, hidden_dim, seq_len]
        # print(pocket_seq.shape, protein_seq.shape)

        compound_feats = torch.mean(self.drug_GVP(xd), dim=1) # [batch_size, hidden_dim] (128, 128)

        protein_feats = torch.mean(self.prot_GVP(xp), dim=1) # [batch_size, hidden_dim] (128, 128)
        seq_feats = torch.mean(self.seq_encoder(protein_seq, pocket_seq), dim=1) # [batch_size, hidden_dim] (128, 128)
        # print(compound_feats.shape, protein_feats.shape, seq_feats.shape)
        combined_feats = torch.cat([compound_feats, protein_feats, seq_feats], dim=-1)
        # combined_feats = self.multi_gating_network(pg=protein_feats, ps=seq_feats, dg=compound_feats)


        # print(protein_feats.shape, compound_feats.shape)

        # ori_compound = compound_feats.permute(0, 2, 1)

        # ori_protein = protein_feats.permute(0, 2, 1)
        #

        # protein_feats = self.prot_mutual_attention(seq_feats.permute(0, 2, 1), protein_feats.permute(0, 2, 1)) # [batch_size, hidden_dim, seq_len]
        # print(protein_feats.shape)
        # protein_feats = torch.cat([seq_feats, protein_feats], dim=-1)
        # print(protein_feats.shape)
        # compound_feats = self.drug_mutual_attention(protein_feats, compound_feats) # [batch_size, hidden_dim, seq_len]
        # # print(compound_feats.permute(0, 2, 1).shape)
        # # print(protein_feats.permute(0, 2, 1).shape)
        # compound_feats = self.drug_fc(compound_feats.permute(0, 2, 1))  # [batch_size, seq_len, hidden_dim]
        # protein_feats = self.prot_fc(protein_feats.permute(0, 2, 1))  # [batch_size, seq_len, hidden_dim]
        # compound_feats+=ori_compound
        # protein_feats+=ori_protein
        #
        # combined_feats, att = self.bcn(compound_feats, protein_feats.permute(0, 2, 1))
        # x = torch.cat([compound_feats, protein_feats], dim=1)

        x = self.mlp(combined_feats)
        return x, compound_feats, protein_feats, seq_feats
