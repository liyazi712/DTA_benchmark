"""
Adapted from
https://github.com/jingraham/neurips19-graph-protein-design
https://github.com/drorlab/gvp-pytorch
"""
import math
import numpy as np
import scipy as sp
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch_geometric
import torch_cluster
from gvp_transformer_pdbbind.constants import LETTER_TO_NUM
from scipy.spatial import KDTree


def pdb_to_graphs(prot_data, params):
    """
    Converts a list of protein dict to a list of torch_geometric graphs.
    Parameters
    ----------
    prot_data : dict
        A list of protein data dict. see format in `featurize_protein_graph()`.
    params : dict
        A dictionary of parameters defined in `featurize_protein_graph()`.
    Returns
    -------
    graphs : dict
        A list of torch_geometric graphs. protein key -> graph
    """
    graphs = {}
    for key, struct in tqdm(prot_data.items(), desc='pdb'):
        graphs[key] = featurize_protein_graph(
            struct, name=key, **params)
    return graphs


def unique_edges(edge_index, knn_idx):
    unique_edges_set = set()
    for i, j in edge_index:
        unique_edges_set.add((min(i, j), max(i, j)))
        unique_edges_set.add((max(i, j), min(i, j)))
    for i, j in knn_idx:
        if (min(i, j), max(i, j)) not in unique_edges_set:
            unique_edges_set.add((min(i, j), max(i, j)))
            unique_edges_set.add((max(i, j), min(i, j)))
    new_edge_index = torch.tensor(list(unique_edges_set), dtype=torch.long)
    return new_edge_index

def featurize_protein_graph(
        protein, name=None,
        num_pos_emb=16, num_rbf=16,        
        contact_cutoff=8.,
):
    """
    Parameters: see comments of DTATask() in dta.py
    """
    with torch.no_grad():  # 确保在构建数据时不计算梯度
        coords = torch.as_tensor(protein['coords'], dtype=torch.float32)  # 将蛋白质的坐标转换为张量
        seq = torch.as_tensor([LETTER_TO_NUM[a] for a in protein['seq']], dtype=torch.long)  # 将氨基酸序列转换为整数序列
        seq_emb = torch.load(protein['embed']) if 'embed' in protein else None  # 如果存在嵌入信息，则加载它

        mask = torch.isfinite(coords.sum(dim=(1,2)))  # 创建一个掩码，标记出有限的坐标值
        coords[~mask] = np.inf  # 将非法坐标（如NaN）替换为无穷大

        X_ca = coords[:, 1]  # 提取Cα原子的坐标作为节点位置
        ca_mask = torch.isfinite(X_ca.sum(dim=(1)))  # 为Cα原子创建掩码
        ca_mask = ca_mask.float()  # 将掩码转换为浮点张量
        ca_mask_2D = torch.unsqueeze(ca_mask, 0) * torch.unsqueeze(ca_mask, 1)  # 扩展掩码为二维
        dX_ca = torch.unsqueeze(X_ca, 0) - torch.unsqueeze(X_ca, 1)  # 计算Cα原子之间的距离
        D_ca = ca_mask_2D * torch.sqrt(torch.sum(dX_ca**2, 2) + 1e-6)  # 计算距离的平方根，并添加小常数以避免除零
        edge_index = torch.nonzero((D_ca < contact_cutoff) & (ca_mask_2D == 1))  # 确定边的索引，根据接触距离阈值

        knn_matrix = np.zeros_like(D_ca)
        tree = KDTree(D_ca)
        k = 3
        # print("X_ca: ", X_ca.shape)
        for i in range(D_ca.shape[0]):
            distances, indices = tree.query(D_ca[i], k=k + 1)  # +1 包括自身
            knn_matrix[i, indices[1:]] = 1  # 排除自身

        knn_edge_index = torch.nonzero(torch.tensor(knn_matrix))

        # print("knn_index: ", knn_edge_index.shape)
        # print("before_edge_index: ", edge_index.shape)

        combined_edge_index = torch.cat((edge_index, knn_edge_index), dim=0)

        # 去重，避免重复的边
        edge_index = torch.unique(combined_edge_index, dim=0)

        # print("after_edge_index: ", edge_index.shape)

        edge_index = edge_index.t().contiguous()  # 转置并连续化边的索引

        O_feature = _local_frame(X_ca, edge_index)  # 计算局部坐标系特征
        pos_embeddings = _positional_embeddings(edge_index, num_embeddings=num_pos_emb)  # 计算位置嵌入特征
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]  # 计算边向量
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf)  # 计算径向基函数特征

        dihedrals = _dihedrals(coords)  # 计算二面角特征
        orientations = _orientations(X_ca)  # 计算主链方向特征
        sidechains = _sidechains(coords)  # 计算侧链特征

        node_s = dihedrals  # 节点的标量特征设置为二面角
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)  # 节点的向量特征是方向和侧链特征的拼接
        edge_s = torch.cat([rbf, O_feature, pos_embeddings], dim=-1)  # 边的标量特征是RBF、局部坐标系和位置嵌入的拼接
        edge_v = _normalize(E_vectors).unsqueeze(-2)  # 边的向量特征是规范化的边向量

        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))  # 将NaN转换为数值

    data = torch_geometric.data.Data(  # 创建图数据对象
        x=X_ca, seq=seq, name=name, coords=coords,
        node_s=node_s, node_v=node_v,
        edge_s=edge_s, edge_v=edge_v,
        edge_index=edge_index, mask=mask,
        seq_emb=seq_emb)
    return data


def _dihedrals(X, eps=1e-7):
    """
    计算蛋白质主链的二面角特征。
    eps 是一个小常数，用于避免在计算时取反余弦值时出现无效的数学运算。
    """
    """
    这个函数首先重新排列输入坐标，然后计算相邻氨基酸残基之间的向量差分，并求得这些向量的规范化表示。
    接着，通过叉乘计算两个相邻氨基酸残基的法向量，进而利用法向量的点积来求得二面角的余弦值。
    之后，通过反余弦函数求得实际的二面角，并通过填充操作扩展角度特征，
    最后将这些角度转换为周期性函数的表示形式，即cos和sin，以便于图神经网络处理。
    """
    X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])  # 重新排列坐标，使得每个氨基酸的三个主链原子（N, CA, C）连续排列
    # print(X.shape)
    dX = X[1:] - X[:-1]  # 计算相邻氨基酸残基之间的原子向量差分

    U = _normalize(dX, dim=-1)  # 规范化差分向量，得到向量的方向
    u_2 = U[:-2]  # 表示 i-2 号氨基酸的向量
    u_1 = U[1:-1]  # 表示 i-1 号氨基酸的向量
    u_0 = U[2:]  # 表示 i 号氨基酸的向量

    # 计算主链上的法向量，用于后续计算二面角
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)  # i-2 和 i-1 号氨基酸向量的叉乘
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)  # i-1 和 i 号氨基酸向量的叉乘

    # 计算两个法向量的点积，得到二面角的余弦值
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)  # 避免在取反余弦时出现数值无效的情况

    # 计算二面角，并将结果扩展为 [BATCH, 3] 形状，其中最后一位是0，用于后续填充
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    # print(D.shape)
    D = F.pad(D, [1, 2])  # 填充，使得D的形状变为 [BATCH, 5]，为每个氨基酸添加额外的二面角特征
    D = torch.reshape(D, [-1, 3])

    # 将二面角转换为周期性表示，即使用cos和sin表示
    # print(D.shape)
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def _positional_embeddings(edge_index,
                            num_embeddings=None,
                            period_range=[2, 1000]):
    """
    生成位置嵌入特征。
    edge_index 是图中边的索引。
    num_embeddings 是要生成的位置嵌入的维度数。
    period_range 是用于控制不同维度频率的最小和最大周期。

    这个函数首先计算图中每条边的两个节点索引的差值，这个差值可以反映节点在序列中的距离。然后，
    使用这个距离信息来计算不同维度的位置嵌入。具体来说，函数计算了一个频率向量，每个频率对应于位置嵌入中的一个维度，
    频率从低到高排列。接着，将每个频率与边的位置差相乘，并计算得到的角的余弦和正弦值。最后，将余弦和正弦值合并为一个向量，
    这个向量就是对应于每条边的位置嵌入。

    这种位置嵌入方法允许模型学习到不同节点间的距离关系，并且能够在处理序列数据时保持不同位置之间的区分度。
    在蛋白质结构分析中，这种编码方式可以帮助模型理解氨基酸残基在蛋白质序列中的相对位置，从而更好地捕捉蛋白质的三维结构特征。
    """
    d = edge_index[0] - edge_index[1]  # 计算边的索引差，反映节点间的位置关系
    frequency = torch.exp(  # 计算不同维度的频率，从低频到高频
        torch.arange(0, num_embeddings, 2, dtype=torch.float32) * -(np.log(10000.0) / num_embeddings))
    # torch.arange(0, num_embeddings, 2, dtype=torch.float32): 这将生成一个从0开始到num_embeddings（不包括）的偶数序列。
    # 因为步长是2，所以生成的序列将只包含偶数索引，这对应于0, 2, 4, ..., num_embeddings-2（如果num_embeddings是偶数）。
    angles = d.unsqueeze(-1) * frequency  # 将边的位置差与频率相乘，扩展为适合做三角函数运算的形状
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)  # 将余弦和正弦值合并为一个向量

    return E


def _orientations(X):
    '''
    这个函数首先计算了相邻氨基酸残基之间的向量，分别指向下一个残基（forward）和上一个残基（backward）。
    这些向量是通过简单的坐标差值来计算的，并且使用 _normalize 函数进行规范化，以确保它们表示的是方向而不是长度。
    接下来，使用 F.pad 对 forward 和 backward 向量进行填充，以确保它们的长度与原始的原子坐标张量 X 相同。
    填充操作是在张量的开始和结束处添加零向量，这样每个氨基酸残基的前后向量就可以与它自己的坐标对应起来。
    最后，使用 torch.cat 将 forward 和 backward 向量沿着一个新的维度（unsqueeze 操作创建这个维度）拼接起来。
    这样，每个氨基酸残基的局部方向信息就被编码为一个形状为 (N, 2, 3) 的张量，其中 N 是残基的数量，
    2 表示每个残基有两个方向向量（前向和后向），3 表示每个向量有三维空间中的 x, y, z 分量。
    这种局部方向信息提供了关于蛋白质主链如何弯曲和折叠的空间信息，这有助于网络理解蛋白质的三维结构特征。
    '''
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(X):
    """
    计算侧链的方向信息。
    X 是一个形状为 (N, 3) 的张量，其中 N 是原子的数量，3 是每个原子的三维坐标。
    这个函数假设 X 包含了每个氨基酸残基的 N (氮), CA (α-碳), C (碳) 原子的坐标。

    这个函数首先从输入坐标张量 X 中提取每个氨基酸残基的 N（氮）、CA（α-碳）、C（碳）原子的坐标。
    然后，计算从 CA 到 C 和从 CA 到 N 的向量，并规范化这些向量以得到方向信息。
    接着，计算 N-CA-C 角的角平分线，即 N 和 C 向量的和的规范化向量。此外，通过叉乘 N 和 C 向量来计算垂直于这个平面的向量，并将结果规范化。
    最后，通过特定的线性组合计算侧链方向向量。这个组合使用了角平分线向量和垂直向量，并且通过权重 -1/sqrt(3) 和 -sqrt(2/3)
    来平衡它们对侧链方向的贡献。这产生了一个表示侧链方向的向量，可以认为是侧链可能伸展的平均方向。
    得到的侧链方向向量 vec 反映了每个氨基酸残基侧链的空间取向，这对于蛋白质结构分析和理解蛋白质功能至关重要。
    在图神经网络中使用这些信息可以帮助模型更准确地捕捉蛋白质的三维结构特征。
    """
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]  # 分别提取 N, CA, C 三个原子的坐标
    c, n = _normalize(c - origin), _normalize(n - origin)  # 计算从 CA 到 C 和从 CA 到 N 的规范化向量
    bisector = _normalize(c + n)  # 计算 N-CA-C 的角平分线（二分线）的规范化向量
    perp = _normalize(torch.cross(c, n))  # 计算 N-CA-C 的垂直向量，即叉乘结果并规范化
    # 计算从角平分线到垂直平面的向量，这代表了侧链的方向
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    """
    根据边的长度 D 计算径向基函数嵌入。D 是一个张量，包含边的长度。D_min 和 D_max 定义了距离嵌入的最大和最小值。
    D_count 是生成的RBF嵌入的维度数。device 是张量所在的设备，'cpu' 或 'gpu'。
    生成的RBF嵌入是一个 (D_count,) 形状的向量，为每条边提供了一个关于其长度的高维特征表示。
    这种表示方式使得图神经网络能够更细致地处理节点间距离信息，捕捉到蛋白质结构中的细微差异。
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)  # 在 D_min 和 D_max 之间均匀生成 D_count 个点
    D_mu = D_mu.view([1, -1])  # 将生成的点重新排布为一行

    D_sigma = (D_max - D_min) / D_count  # 计算标准差，即每个RBF中心点的间隔
    D_expand = torch.unsqueeze(D, -1)  # 扩展D为适合做传播运算的形状

    # 计算高斯RBF核，即每个距离 D 与中心点 D_mu 的差的平方标准化后，再取负指数
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def _local_frame(X, edge_index, eps=1e-6):
    """
    计算局部坐标系特征。
    X 是包含原子坐标的张量。
    edge_index 定义了图中的边。
    eps 是一个小常数，用于数值稳定性。
    """
    """
    这个函数首先计算了每条边的方向向量，并规范化这些向量以得到主链的方向。然后，通过叉乘计算出主链的法向量。
    接着，使用这些向量来构建每个原子的局部坐标系。然后，计算了每条边在局部坐标系下的单位向量，
    并通过旋转矩阵和四元数来描述原子间的相对旋转。最后，将局部坐标系下的方向和四元数特征合并，形成每条边的局部坐标系特征向量。
    这些特征对于图神经网络来说非常重要，因为它们提供了原子间相对空间位置和旋转的信息，这有助于网络更好地理解蛋白质的三维结构。
    """
    dX = X[edge_index[1]] - X[edge_index[0]]  # 计算边的方向向量，从节点0到节点1
    U = _normalize(dX, dim=-1)  # 规范化方向向量，得到主链的方向
    u_2 = U[:-2]  # i-2 号原子的方向向量
    u_1 = U[1:-1]  # i-1 号原子的方向向量
    u_0 = U[2:]  # i 号原子的方向向量

    # 计算主链上的法向量
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # 计算骨架的切线方向
    o_1 = _normalize(u_2 - u_1, dim=-1)
    O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 1)
    O = F.pad(O, (0, 0, 0, 0, 1, 2), 'constant', 0)  # 填充O以匹配X的形状

    # 计算每条边的单位向量
    dX = X[edge_index[0]] - X[edge_index[1]]
    dX = _normalize(dX, dim=-1)
    dU = torch.bmm(O[edge_index[0]], dX.unsqueeze(2)).squeeze(2)  # 将O与dX结合，得到局部坐标系下的方向
    # bmm全称为 "Batch Matrix Multiply"，即批量矩阵乘法。这个函数用于对两个输入的张量进行批量矩阵乘操作，通常用于处理三维张量。
    R = torch.bmm(O[edge_index[0]].transpose(-1,-2), O[edge_index[1]])  # 计算旋转矩阵，R:（B,3,3）
    Q = _quaternions(R)  # 从旋转矩阵计算四元数特征

    # 将局部坐标系下的方向和四元数特征合并
    O_features = torch.cat((dU, Q), dim=-1)
    return O_features

def _quaternions(R):
    # Simple Wikipedia version
    # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
    """
    从旋转矩阵 R 计算四元数。
    R 是一个形状为 (B, 3, 3) 的张量，其中 B 是批次大小。

    这个函数首先从旋转矩阵 R 中提取对角线元素，并分别解绑为三个独立的向量 Rxx, Ryy, Rzz。然后，
    使用这些对角线元素计算四元数的 x, y, z 分量的幅度。接着，通过比较旋转矩阵中特定元素对的值来确定四元数分量的符号。
    之后，计算四元数的 w 分量，这里使用 torch.sqrt 和 F.relu 来确保 w 的值是非负的。将 x, y, z, w 分量合并为一个四元素的向量 Q，
    最后通过 F.normalize 对四元数进行规范化，确保它的长度为 1，这是四元数的一个基本性质。
    规范化的四元数可以用于表示三维空间中的旋转，并且在物理和计算机图形学中广泛应用。
    在生物分子结构分析中，四元数可以用来描述氨基酸残基之间的相对旋转。
    """
    diag = torch.diagonal(R, dim1=-2, dim2=-1)  # 从旋转矩阵中提取对角线元素
    Rxx, Ryy, Rzz = diag.unbind(-1)  # 对角线元素分别解绑为三个独立的向量

    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
        Rxx - Ryy - Rzz,
        -Rxx + Ryy - Rzz,
        -Rxx - Ryy + Rzz
    ], -1)))  # 计算每个四元数分量的幅度
    _R = lambda i, j: R[:, i, j]  # 创建一个获取 R 中特定元素的函数

    signs = torch.sign(torch.stack([
        _R(2, 1) - _R(1, 2),
        _R(0, 2) - _R(2, 0),
        _R(1, 0) - _R(0, 1)
    ], -1))  # 计算四元数分量的符号

    xyz = signs * magnitudes  # 计算四元数的 x, y, z 分量
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.  # 计算四元数的 w 分量，使用 relu 保证 w 非负

    Q = torch.cat((xyz, w), -1)  # 将四元数的 x, y, z, w 分量合并为一个向量
    Q = F.normalize(Q, dim=-1)  # 规范化四元数，使其长度为 1
    return Q

