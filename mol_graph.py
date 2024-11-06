import rdkit.Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import torch_geometric
import torch_cluster

from gvp_transformer_pdbbind.constants import ATOM_VOCAB
from gvp_transformer_pdbbind.pdb_graph import _rbf, _normalize
from scipy.spatial import KDTree


def onehot_encoder(a=None, alphabet=None, default=None, drop_first=False):
    '''
    Parameters
    ----------
    a: array of numerical value of categorical feature classes.
    alphabet: valid values of feature classes.
    default: default class if out of alphabet.
    Returns
    -------
    A 2-D one-hot array with size |x| * |alphabet|
    '''
    # replace out-of-vocabulary classes
    alphabet_set = set(alphabet)
    a = [x if x in alphabet_set else default for x in a]

    # cast to category to force class not present
    a = pd.Categorical(a, categories=alphabet)

    onehot = pd.get_dummies(pd.Series(a), columns=alphabet, drop_first=drop_first)
    return onehot.values


def _build_atom_feature(mol):
    # dim: 44 + 7 + 7 + 7 + 1
    feature_alphabet = {
        # (alphabet, default value)
        'GetSymbol': (ATOM_VOCAB, 'unk'),
        'GetDegree': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetTotalNumHs': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetImplicitValence': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetIsAromatic': ([0, 1], 1)
    }

    atom_feature = None
    for attr in ['GetSymbol', 'GetDegree', 'GetTotalNumHs',
                'GetImplicitValence', 'GetIsAromatic']:
        feature = [getattr(atom, attr)() for atom in mol.GetAtoms()]
        feature = onehot_encoder(feature,
                    alphabet=feature_alphabet[attr][0],
                    default=feature_alphabet[attr][1],
                    drop_first=(attr in ['GetIsAromatic']) # binary-class feature
                )
        atom_feature = feature if atom_feature is None else np.concatenate((atom_feature, feature), axis=1)
    atom_feature = atom_feature.astype(np.float32)
    return atom_feature

def knn_edge_index(D_ca, K):
    N = D_ca.size(0)  # 节点数量
    edge_index = []  # 初始化边索引列表

    for i in range(N):
        dist_i = D_ca[i]
        dist_i[i] = torch.finfo(dist_i.dtype).max + 1
        # 找到 K 个最近邻的索引，排除自身
        k_nearest = torch.argsort(dist_i)[:K]
        for j in k_nearest:
            edge_index.append([i, j])
            edge_index.append([j, i])
    return edge_index

def unique_edges(edge_index, knn_idx):
    unique_edges_set = set()
    for i, j in edge_index.t():
        unique_edges_set.add((min(i, j), max(i, j)))
        unique_edges_set.add((max(i, j), min(i, j)))
    for i, j in knn_idx:
        if (min(i, j), max(i, j)) not in unique_edges_set:
            unique_edges_set.add((min(i, j), max(i, j)))
            unique_edges_set.add((max(i, j), min(i, j)))
    new_edge_index = torch.tensor(list(unique_edges_set), dtype=torch.long)
    return new_edge_index

def _build_edge_feature(coords, edge_index, D_max=4.5, num_rbf=16):
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_max=D_max, D_count=num_rbf)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


def sdf_to_graphs(data_list):
    """
    Parameters
    ----------
    data_list: dict, drug key -> sdf file path
    Returns
    -------
    graphs : dict
        A list of torch_geometric graphs. drug key -> graph
    """
    graphs = {}
    for key, value in tqdm(data_list.items(), desc='sdf'):
        graphs[key] = featurize_drug(coords=value['coords'], atom_feature=value['atoms_feature'], name=key)
    return graphs


def featurize_drug(coords, atom_feature, name=None, edge_cutoff=4.5, num_rbf=16):
    """
    Parameters
    ----------
    sdf_path: str
        Path to sdf file
    name: str
        Name of drug
    Returns
    -------
    graph: torch_geometric.data.Data
        A torch_geometric graph
    """
    with torch.no_grad():
        coords = torch.as_tensor(coords, dtype=torch.float32)
        atom_feature = torch.as_tensor(atom_feature, dtype=torch.float32)
        edge_index = torch_cluster.radius_graph(coords, r=edge_cutoff)

        dX_ca = torch.unsqueeze(coords, 0) - torch.unsqueeze(coords, 1)  # 计算Cα原子之间的距离
        D_ca = torch.sqrt(torch.sum(dX_ca ** 2, 2) + 1e-6)  # 计算距离的平方根，并添加小常数以避免除零

        # knn_matrix = np.zeros_like(D_ca)
        # tree = KDTree(D_ca)
        # k = 3
        # # print("X_ca: ", X_ca.shape)
        # for i in range(D_ca.shape[0]):
        #     distances, indices = tree.query(D_ca[i], k=k + 1)  # +1 包括自身
        #     knn_matrix[i, indices[1:]] = 1  # 排除自身
        # knn_edge_index = torch.nonzero(torch.tensor(knn_matrix))
        # combined_edge_index = torch.cat((edge_index, knn_edge_index), dim=0)
        # edge_index = torch.unique(combined_edge_index, dim=0)

        # knn_idx = knn_edge_index(D_ca, 3)
        # edge_index = unique_edges(edge_index, knn_idx)
        # edge_index = edge_index.t().contiguous()  # 转置并连续化边的索引

    node_s = atom_feature
    node_v = coords.unsqueeze(1)
    edge_s, edge_v = _build_edge_feature(
        coords, edge_index, D_max=edge_cutoff, num_rbf=num_rbf)

    data = torch_geometric.data.Data(
        x=coords, edge_index=edge_index, name=name,
        node_v=node_v, node_s=node_s, edge_v=edge_v, edge_s=edge_s)
    return data

