"""
Drug-target binding affinity datasets
"""
import json
from functools import partial
from pathlib import Path

import dgl
import torch
from torch_geometric.data import Batch
# from torch_geometric.utils import batch
from .pdbbind_utils import data_from_index
import numpy as np
import pandas as pd
import torch.utils.data as data
from gvp_transformer_pdbbind import pdb_graph, mol_graph

class DTA(data.Dataset):
    """
    Base class for loading drug-target binding affinity datasets.
    Adapted from: https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py
    """
    def __init__(self, data=None, data_list=None, onthefly=False,
                prot_featurize_fn=None, drug_featurize_fn=None):

        super(DTA, self).__init__()
        self.data_list = data_list
        self.onthefly = onthefly
        if onthefly:
            assert prot_featurize_fn is not None, 'prot_featurize_fn must be provided'
            assert drug_featurize_fn is not None, 'drug_featurize_fn must be provided'
        self.prot_featurize_fn = prot_featurize_fn
        self.drug_featurize_fn = drug_featurize_fn
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.onthefly:
            drug = self.drug_featurize_fn(
                self.data_list[idx]['drug'],
                name=self.data_list[idx]['drug_name']
            )
            prot = self.prot_featurize_fn(
                self.data_list[idx]['protein'],
                name=self.data_list[idx]['protein_name']
            )
        else:
            drug = self.data_list[idx]['drug_graph']
            prot = self.data_list[idx]['protein_graph']
            full_seq = self.data_list[idx]['full_sequence']
            poc_seq = self.data_list[idx]['pocket_sequence']
            y = self.data_list[idx]['y']
        # item = {'drug': drug, 'protein': prot, 'y': y}
        return drug, prot, full_seq, poc_seq, float(y)

    def collate(self, sample):
        batch_size = len(sample)
        # print(f'batch_size: {batch_size}')
        compound_graph, protein_graph, full_seq, poc_seq, label = map(list, zip(*sample))
        compound_graph = Batch.from_data_list(compound_graph)
        protein_graph = Batch.from_data_list(protein_graph)
        full_seq = torch.tensor(np.array(full_seq)).long()
        poc_seq = torch.tensor(np.array(poc_seq)).long()
        # print(full_seq, poc_seq)
        label = torch.FloatTensor(label)

        item = {'drug': compound_graph, 'protein': protein_graph, 'full_seq': full_seq, 'poc_seq': poc_seq, 'y': label}
        return item

class DTATask(object):
    """
    Drug-target binding task (e.g., pdbbind_v2021 or ).
    """
    def __init__(self,
            task_name=None,
            split_method = 'new_protein',
            train_data=None, valid_data=None, test_data=None,
            train_pdb_data=None, valid_pdb_data=None, test_pdb_data=None,
            train_sdf_data=None, valid_sdf_data=None, test_sdf_data=None,
            max_seq_len=None,
            emb_dir=None,
            drug_sdf_dir=None,
            num_pos_emb=16, num_rbf=16,
            contact_cutoff=8., onthefly=False
        ):

        self.task_name = task_name
        self.train_pdb_data = train_pdb_data
        self.valid_pdb_data = valid_pdb_data
        self.test_pdb_data = test_pdb_data
        self.train_sdf_data = train_sdf_data
        self.valid_sdf_data = valid_sdf_data
        self.test_sdf_data = test_sdf_data
        self.emb_dir = emb_dir
        self.split_method = split_method
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.max_seq_len = max_seq_len

        self.prot_featurize_params = dict(
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            contact_cutoff=contact_cutoff)        
        self.drug_sdf_dir = drug_sdf_dir        
        self._prot2pdb = None
        self._pdb_graph_db = None        
        self._drug2sdf_file = None
        self._drug_sdf_db = None
        self.onthefly = onthefly

    def _format_pdb_entry(self, _data):
        _coords = _data["coords"]
        entry = {
            "pdb_id": _data["pdb_id"],
            "seq": _data["seq"],
            "coords": list(zip(_coords["N"], _coords["CA"], _coords["C"], _coords["O"])),
        }
        if self.emb_dir is not None:
            embed_file = f"{_data['PDB_id']}.{_data['chain']}.pt"
            entry["embed"] = f"{self.emb_dir}/{embed_file}"
        return entry

    @property
    def pdb_graph_db(self):
        if self._pdb_graph_db is None:
            pdbid_entry = {}
            combined_pdb_data = {**self.train_pdb_data, **self.valid_pdb_data, **self.test_pdb_data}
            for pdb_id, protein_structure_info in combined_pdb_data.items():
                pdbid_entry[pdb_id] = self._format_pdb_entry(protein_structure_info)
            self._pdb_graph_db = pdb_graph.pdb_to_graphs(pdbid_entry, self.prot_featurize_params)
        return self._pdb_graph_db

    @property
    def drug2sdf_file(self):
        if self._drug2sdf_file is None:            
            drug2sdf_file = {f.stem[:4] : str(f) for f in Path(self.drug_sdf_dir).glob('*.sdf')}
            # Convert str keys to int for Davis
            # if self.task_name == 'DAVIS' and all([k.isdigit() for k in drug2sdf_file.keys()]):
            drug2sdf_file = {str(k) : v for k, v in drug2sdf_file.items()}
            self._drug2sdf_file = drug2sdf_file
        return self._drug2sdf_file
    @property
    def drug_sdf_db(self):
        if self._drug_sdf_db is None:
            # sdfid_entry = {}
            # for pdb_id, mol_structure_info in self.sdf_data.items():
            #     sdfid_entry[pdb_id] = mol_structure_info
            # self._drug_sdf_db = mol_graph.sdf_to_graphs(self.sdf_data)

            # 汇总 train、valid 和 test 的 SDF 数据
            combined_sdf_data = {**self.train_sdf_data, **self.valid_sdf_data, **self.test_sdf_data}

            # 将汇总的 SDF 数据转换为图
            self._drug_sdf_db = mol_graph.sdf_to_graphs(combined_sdf_data)
        return self._drug_sdf_db

    def label_seq(self, line, max_seq_len):
        protein_char = {'<MASK>': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4,
                        'F': 5, 'G': 6, 'H': 7, 'K': 8,
                        'I': 9, 'L': 10, 'M': 11, 'N': 12,
                        'P': 13, 'Q': 14, 'R': 15, 'S': 16,
                        'T': 17, 'V': 18, 'Y': 19, 'W': 20, 'X':21}

        label = np.zeros(max_seq_len)
        for i, lab in enumerate(line[:max_seq_len]):
            label[i] = protein_char[lab]
        return label

    def position_seq(self, seq, position):
        res = ['<MASK>'] * len(seq)
        for i in position:
            res[i - 1] = seq[i - 1]
        return res


    def build_data(self, data, onthefly=False):
        records = data.to_dict('records')
        # print(records)
        data_list = []
        for entry in records:
            # print(entry)
            pdb_id = entry['PDBname']
            if onthefly:
                pf = self.prot2pdb[pdb_id]
                df = self.drug2sdf_file[pdb_id]
            else:                
                pf = self.pdb_graph_db[pdb_id]
                df = self.drug_sdf_db[pdb_id]

            seq = entry['Sequence']
            position = eval(entry['Position'])
            pocket = self.position_seq(seq, position)
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            seq_encode = self.label_seq(seq, self.max_seq_len)
            pocket_encode = self.label_seq(pocket, self.max_seq_len)
            # seq_encode = torch.tensor(self.label_seq(seq, self.max_seq_len), device=device).long()
            # pocket_encode = torch.tensor(self.label_seq(pocket, self.max_seq_len), device=device).long()

            data_list.append({'drug_graph': df, 'protein_graph': pf, 'full_sequence': seq_encode,
                 'pocket_sequence': pocket_encode, 'y': entry['label'], 'pdb_name': pdb_id})
        if onthefly:
            prot_featurize_fn = partial(
                pdb_graph.featurize_protein_graph,
                **self.prot_featurize_params)            
            drug_featurize_fn = mol_graph.featurize_drug
        else:
            prot_featurize_fn, drug_featurize_fn = None, None
        data = DTA(data=data, data_list=data_list, onthefly=onthefly,
            prot_featurize_fn=prot_featurize_fn, drug_featurize_fn=drug_featurize_fn)
        return data

    def get_split(self, onthefly=None):
        onthefly = onthefly or self.onthefly
        split_df = {'train': self.train_data, 'valid': self.valid_data, 'test': self.test_data}
        split_data = {}
        split_data['train'] = self.build_data(self.train_data, onthefly=onthefly)
        split_data['valid'] = self.build_data(self.valid_data, onthefly=onthefly)
        split_data['test'] = self.build_data(self.test_data, onthefly=onthefly)

        return split_data, split_df

class pdbbind_v2016(DTATask):
    """
    pdbbind_v2016 drug-target interaction dataset
    """
    def __init__(self,
            train_path='./create_dataset/PDBBindv2016/last_train_2016.csv',
            valid_path='./create_dataset/PDBBindv2016/last_valid_2016.csv',
            test_path='./create_dataset/PDBBindv2016/last_test_2016.csv',
            train_pdb_json='./create_dataset/PDBBindv2016/pocket_structures_train.json',
            valid_pdb_json='./create_dataset/PDBBindv2016/pocket_structures_valid.json',
            test_pdb_json='./create_dataset/PDBBindv2016/pocket_structures_test.json',
            train_sdf_json='./create_dataset/PDBBindv2016/mol_structures_train.json',
            valid_sdf_json='./create_dataset/PDBBindv2016/mol_structures_valid.json',
            test_sdf_json='./create_dataset/PDBBindv2016/mol_structures_test.json',
            num_pos_emb=16, num_rbf=16, max_seq_len=800, # 1024
            contact_cutoff=8.,
            drug_sdf_dir='./create_dataset/PDBBindv2016/pdbbind_v2016_mol3d_sdf',
            onthefly=False
        ):
        train_data = pd.read_csv(train_path)
        valid_data = pd.read_csv(valid_path)
        test_data = pd.read_csv(test_path)
        train_pdb_data = json.load(open(train_pdb_json, 'r'))
        valid_pdb_data = json.load(open(valid_pdb_json, 'r'))
        test_pdb_data = json.load(open(test_pdb_json, 'r'))
        train_sdf_data = json.load(open(train_sdf_json, 'r'))
        valid_sdf_data = json.load(open(valid_sdf_json, 'r'))
        test_sdf_data = json.load(open(test_sdf_json, 'r'))
        super(pdbbind_v2016, self).__init__(
            task_name='pdbbind_v2016',
            train_data=train_data, valid_data=valid_data, test_data=test_data,
            train_pdb_data=train_pdb_data, valid_pdb_data=valid_pdb_data, test_pdb_data=test_pdb_data,
            train_sdf_data=train_sdf_data, valid_sdf_data=valid_sdf_data, test_sdf_data=test_sdf_data,
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            max_seq_len=max_seq_len,
            contact_cutoff=contact_cutoff,
            drug_sdf_dir=drug_sdf_dir,
            onthefly=onthefly
            )


class pdbbind_v2020(DTATask):
    """
    pdbbind_v20120 drug-target interaction dataset
    """
    def __init__(self,
            train_path='./create_dataset/PDBBindv2020/last_train_2020.csv',
            valid_path='./create_dataset/PDBBindv2020/last_valid_2020.csv',
            test_path='./create_dataset/PDBBindv2020/last_test_2020.csv',
            train_pdb_json='./create_dataset/PDBBindv2020/pocket_structures_train.json',
            valid_pdb_json='./create_dataset/PDBBindv2020/pocket_structures_valid.json',
            test_pdb_json='./create_dataset/PDBBindv2020/pocket_structures_test.json',
            train_sdf_json='./create_dataset/PDBBindv2020/mol_structures_train.json',
            valid_sdf_json='./create_dataset/PDBBindv2020/mol_structures_valid.json',
            test_sdf_json='./create_dataset/PDBBindv2020/mol_structures_test.json',
            num_pos_emb=16, num_rbf=16, max_seq_len=800,
            contact_cutoff=8.,
            drug_sdf_dir='./create_dataset/PDBBindv2020/pdbbind_v2020_mol3d_sdf',
            onthefly=False
        ):
        train_data = pd.read_csv(train_path)
        valid_data = pd.read_csv(valid_path)
        test_data = pd.read_csv(test_path)
        train_pdb_data = json.load(open(train_pdb_json, 'r'))
        valid_pdb_data = json.load(open(valid_pdb_json, 'r'))
        test_pdb_data = json.load(open(test_pdb_json, 'r'))
        train_sdf_data = json.load(open(train_sdf_json, 'r'))
        valid_sdf_data = json.load(open(valid_sdf_json, 'r'))
        test_sdf_data = json.load(open(test_sdf_json, 'r'))
        super(pdbbind_v2020, self).__init__(
            task_name='pdbbind_v2020',
            train_data=train_data, valid_data=valid_data, test_data=test_data,
            train_pdb_data=train_pdb_data, valid_pdb_data=valid_pdb_data, test_pdb_data=test_pdb_data,
            train_sdf_data=train_sdf_data, valid_sdf_data=valid_sdf_data, test_sdf_data=test_sdf_data,
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            max_seq_len=max_seq_len,
            contact_cutoff=contact_cutoff,
            drug_sdf_dir=drug_sdf_dir,
            onthefly=onthefly
            )

class pdbbind_v2021_time(DTATask):
    """
    pdbbind_v2021_time drug-target interaction dataset
    """
    def __init__(self,
            train_path='./create_dataset/PDBBindv2021/PDBBindv2021_time/train_2021.csv',
            valid_path='./create_dataset/PDBBindv2021/PDBBindv2021_time/valid_2021.csv',
            test_path='./create_dataset/PDBBindv2021/PDBBindv2021_time/test_2021.csv',
            train_pdb_json='./create_dataset/PDBBindv2021/PDBBindv2021_time/pocket_structures_train.json',
            valid_pdb_json='./create_dataset/PDBBindv2021/PDBBindv2021_time/pocket_structures_valid.json',
            test_pdb_json='./create_dataset/PDBBindv2021/PDBBindv2021_time/pocket_structures_test.json',
            train_sdf_json='./create_dataset/PDBBindv2021/PDBBindv2021_time/mol_structures_train.json',
            valid_sdf_json='./create_dataset/PDBBindv2021/PDBBindv2021_time/mol_structures_valid.json',
            test_sdf_json='./create_dataset/PDBBindv2021/PDBBindv2021_time/mol_structures_test.json',
            num_pos_emb=16, num_rbf=16, max_seq_len=800,
            contact_cutoff=8.,
            drug_sdf_dir='./create_dataset/PDBBindv2021/pdbbind_v2021_mol3d_sdf',
            onthefly=False
        ):
        train_data = pd.read_csv(train_path)
        valid_data = pd.read_csv(valid_path)
        test_data = pd.read_csv(test_path)
        train_pdb_data = json.load(open(train_pdb_json, 'r'))
        valid_pdb_data = json.load(open(valid_pdb_json, 'r'))
        test_pdb_data = json.load(open(test_pdb_json, 'r'))
        train_sdf_data = json.load(open(train_sdf_json, 'r'))
        valid_sdf_data = json.load(open(valid_sdf_json, 'r'))
        test_sdf_data = json.load(open(test_sdf_json, 'r'))
        super(pdbbind_v2021_time, self).__init__(
            task_name='pdbbind_v2021_time',
            train_data=train_data, valid_data=valid_data, test_data=test_data,
            train_pdb_data=train_pdb_data, valid_pdb_data=valid_pdb_data, test_pdb_data=test_pdb_data,
            train_sdf_data=train_sdf_data, valid_sdf_data=valid_sdf_data, test_sdf_data=test_sdf_data,
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            max_seq_len=max_seq_len,
            contact_cutoff=contact_cutoff,
            drug_sdf_dir=drug_sdf_dir,
            onthefly=onthefly
            )

class pdbbind_v2021_similarity(DTATask):
    """
    pdbbind_v2021_similarity drug-target interaction dataset
    """
    def __init__(self,
            train_path=f'./create_dataset/PDBBindv2021/PDBBindv2021_similarity/new_new/train_0.5.csv',
            valid_path='./create_dataset/PDBBindv2021/PDBBindv2021_similarity/new_new/valid_0.5.csv',
            test_path='./create_dataset/PDBBindv2021/PDBBindv2021_similarity/new_new/test_0.5.csv',
            train_pdb_json='./create_dataset/PDBBindv2021/PDBBindv2021_similarity/new_new/pocket_structures_train_0.5.json',
            valid_pdb_json='./create_dataset/PDBBindv2021/PDBBindv2021_similarity/new_new/pocket_structures_valid_0.5.json',
            test_pdb_json='./create_dataset/PDBBindv2021/PDBBindv2021_similarity/new_new/pocket_structures_test_0.5.json',
            train_sdf_json='./create_dataset/PDBBindv2021/PDBBindv2021_similarity/new_new/mol_structures_train_0.5.json',
            valid_sdf_json='./create_dataset/PDBBindv2021/PDBBindv2021_similarity/new_new/mol_structures_valid_0.5.json',
            test_sdf_json='./create_dataset/PDBBindv2021/PDBBindv2021_similarity/new_new/mol_structures_test_0.5.json',
            num_pos_emb=16, num_rbf=16, max_seq_len=800,
            contact_cutoff=8.,
            drug_sdf_dir='./create_dataset/PDBBindv2021/pdbbind_v2021_mol3d_sdf',
            onthefly=False
        ):
        train_data = pd.read_csv(train_path)
        valid_data = pd.read_csv(valid_path)
        test_data = pd.read_csv(test_path)
        train_pdb_data = json.load(open(train_pdb_json, 'r'))
        valid_pdb_data = json.load(open(valid_pdb_json, 'r'))
        test_pdb_data = json.load(open(test_pdb_json, 'r'))
        train_sdf_data = json.load(open(train_sdf_json, 'r'))
        valid_sdf_data = json.load(open(valid_sdf_json, 'r'))
        test_sdf_data = json.load(open(test_sdf_json, 'r'))
        super(pdbbind_v2021_similarity, self).__init__(
            task_name='pdbbind_v2021_similarity',
            train_data=train_data, valid_data=valid_data, test_data=test_data,
            train_pdb_data=train_pdb_data, valid_pdb_data=valid_pdb_data, test_pdb_data=test_pdb_data,
            train_sdf_data=train_sdf_data, valid_sdf_data=valid_sdf_data, test_sdf_data=test_sdf_data,
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            max_seq_len=max_seq_len,
            contact_cutoff=contact_cutoff,
            drug_sdf_dir=drug_sdf_dir,
            onthefly=onthefly
            )

