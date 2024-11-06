# import torch
# from equiformer_pytorch import Equiformer
#
# model = Equiformer(
#     num_tokens = 28,
#     dim = (4, 4, 4),               # dimensions per type, ascending, length must match number of degrees (num_degrees)
#     dim_head = (4, 4, 4),          # dimension per attention head
#     heads = (2, 2, 2),             # number of attention heads
#     num_linear_attn_heads = 0,     # number of global linear attention heads, can see all the neighbors
#     num_degrees = 3,               # number of degrees
#     depth = 2,                     # depth of equivariant transformer
#     attend_self = True,            # attending to self or not
#     reduce_dim_out = True,         # whether to reduce out to dimension of 1, say for predicting new coordinates for type 1 features
#     l2_dist_attention = False      # set to False to try out MLP attention
# )
# print(model)
#
#
# atoms = torch.randint(0, 28, (2, 32)) # 28种原子类型，2个样本，32个原子
# # bonds = torch.randint(0, 4, (2, 32, 32)) # 4种键类型，2个样本，32*32的键矩阵，表示每两个原子间的键的类型
# coors = torch.randn(2, 32, 3) # 正态分布，2个样本，32个原子，3个坐标（xyz）
# mask  = torch.ones(2, 32).bool() # 判断原子是否为有效原子，2个样本，32个原子
# # print('atoms:', atoms, 'bonds:', bonds, 'coors:', coors, 'mask:', mask )
#
# out = model(atoms, coors)
#
# out.type0 # (2, 32)
# out.type1 # (2, 32, 3)
# print("out1:", out.type0.shape, "out2:", out.type1.shape)


import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, edge_index, edge_weights=None):
        # node_features: (num_nodes, embed_dim)
        # edge_index: (2, num_edges) tensor containing the indices of the source and target nodes of the edges
        # edge_weights: (num_edges,) tensor containing the weights of the edges

        # Apply self-attention
        attention_output = self.attention(node_features, node_features, node_features, edge_index, edge_weights)
        # Layer normalization and residual connection
        x = self.norm1(node_features + self.dropout(attention_output))

        # Apply feed-forward network
        ff_output = self.feed_forward(x)
        # Second layer normalization and residual connection
        x = self.norm2(x + self.dropout(ff_output))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, edge_index, edge_weights=None):
        # Register hooks to apply edge weights if provided
        if edge_weights is not None:
            edge_weights = edge_weights.view(-1, 1) * self.head_dim
            self.k_proj.register_forward_hook(self.apply_edge_weights_hook(edge_weights))

        # Project queries, keys, values
        queries = self.q_proj(query)
        keys = self.k_proj(key)
        values = self.v_proj(value)

        # Split into self.num_heads
        batch_size, num_nodes, dim = queries.shape
        queries = queries.view(batch_size, num_nodes, self.num_heads, dim // self.num_heads).transpose(1, 2)
        keys = keys.view(batch_size, num_nodes, self.num_heads, dim // self.num_heads).transpose(1, 2)
        values = values.view(batch_size, num_nodes, self.num_heads, dim // self.num_heads).transpose(1, 2)

        # Calculate attention scores
        attention_scores = torch.einsum("bnht,bnht->bnht", [queries, keys])
        if edge_weights is not None:
            attention_scores = attention_scores * edge_weights

        # Apply scaling
        attention_scores = attention_scores / (self.head_dim ** 0.5)

        # Apply softmax and dropout
        attention_probs = self.dropout(F.softmax(attention_scores, dim=-1))

        # Aggregate values using attention probabilities
        attention_output = torch.einsum("bnht,bnht->bnh", [attention_probs, values]).transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, num_nodes, self.embed_dim)

        return attention_output

    def apply_edge_weights_hook(self, edge_weights):
        def hook(module, input, output):
            # Scale the output of the key projection by the edge weights
            return output * edge_weights
        return hook

class GraphTransformer(nn.Module):
    def __init__(self, num_nodes, embed_dim, num_layers, num_heads, dropout=0.1):
        super(GraphTransformer, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, edge_index, edge_weights=None):
        x = self.embedding(node_features)
        for layer in self.layers:
            x = layer(x, edge_index, edge_weights)
        return x

# Example usage:
# Define the number of nodes, embedding dimensions, layers, and heads
num_nodes = 10
embed_dim = 64
num_layers = 2
num_heads = 4

# Instantiate the GraphTransformer model
model = GraphTransformer(num_nodes, embed_dim, num_layers, num_heads)

# Generate some random node features and edge indices
node_features = torch.rand((num_nodes,))
edge_index = torch.randint(0, num_nodes, (2, 20))  # 20 random edges

# Forward pass
output = model(node_features, edge_index)