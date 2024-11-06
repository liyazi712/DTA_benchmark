import numpy as np
import pickle
from sklearn.model_selection import KFold
import csv
import torch
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(100)

#padding functions
def pad_label(arr_list, ref_list):
	N = ref_list.shape[1]
	a = np.zeros((len(arr_list), N))
	for i, arr in enumerate(arr_list):
		n = len(arr)
		a[i,0:n] = arr
	return a

#embedding selection function
def add_index(input_array, ebd_size):
	batch_size, n_vertex, n_nbs = np.shape(input_array)
	# add_idx = np.array(range(0,(ebd_size)*batch_size,ebd_size)*(n_nbs*n_vertex))
	add_idx = np.arange(0, (ebd_size * batch_size), ebd_size) * (n_nbs * n_vertex)
	add_idx = np.transpose(add_idx.reshape(-1,batch_size))
	add_idx = add_idx.reshape(-1)
	new_array = input_array.reshape(-1)+add_idx
	return new_array

# load data
def data_from_index(data_pack, idx_list):
	pdbid_list = []
	pid_list = []
	cid_list = []
	smile_list = []
	seq_list = []
	aff_label_list = []
	for idx in idx_list:
		pdbid_list.append(data_pack.iloc[idx, 1])
		pid_list.append(data_pack.iloc[idx, 2])
		cid_list.append(data_pack.iloc[idx, 3])
		smile_list.append(data_pack.iloc[idx, 4])
		seq_list.append(data_pack.iloc[idx, 5])
		aff_label_list.append(data_pack.iloc[idx, 6])

		# pdbid_list.append(data_pack[idx][1])
		# pid_list.append(data_pack[idx][2])
		# cid_list.append(data_pack[idx][3])
		# smile_list.append(data_pack[idx][4])
		# seq_list.append(data_pack[idx][5])
		# aff_label_list.append(data_pack[idx][6])
	# pdbid, pid, cid, smile, seq, aff_label = [data_pack[i][idx_list] for i in range(1,7)]
	# aff_label = data_pack[6][idx_list].astype(float).reshape(-1,1)
	return [pdbid_list, pid_list, cid_list, smile_list, seq_list, aff_label_list]


def split_train_test_clusters(measure, clu_thre, n_fold):
	# load cluster dict
	cluster_path = './'
	with open(cluster_path+measure+'_compound_cluster_dict_'+str(clu_thre), 'rb') as f:
		C_cluster_dict = pickle.load(f)
	with open(cluster_path+measure+'_protein_cluster_dict_'+str(clu_thre), 'rb') as f:
		P_cluster_dict = pickle.load(f)
	
	C_cluster_set = set(list(C_cluster_dict.values()))
	P_cluster_set = set(list(P_cluster_dict.values()))
	C_cluster_list = np.array(list(C_cluster_set))
	P_cluster_list = np.array(list(P_cluster_set))
	np.random.shuffle(C_cluster_list)
	np.random.shuffle(P_cluster_list)
	# n-fold split
	# c_kf = KFold(len(C_cluster_list), n_fold, shuffle=True)
	# p_kf = KFold(len(P_cluster_list), n_fold, shuffle=True)
	c_kf = KFold(n_fold,shuffle=True,random_state=42)
	p_kf = KFold(n_fold,shuffle=True,random_state=42)
	c_train_clusters, c_test_clusters = [], []
	for train_idx, test_idx in c_kf.split(C_cluster_list):
		c_train_clusters.append(C_cluster_list[train_idx])
		c_test_clusters.append(C_cluster_list[test_idx])
	p_train_clusters, p_test_clusters = [], []
	for train_idx, test_idx in p_kf.split(P_cluster_list):
		p_train_clusters.append(P_cluster_list[train_idx])
		p_test_clusters.append(P_cluster_list[test_idx])
	
	
	#pair_kf = KFold(n_fold,shuffle=True)
	pair_list = []
	for i_c in C_cluster_list:
		for i_p in P_cluster_list:
			pair_list.append('c'+str(i_c)+'p'+str(i_p))
	pair_list = np.array(pair_list)
	np.random.shuffle(pair_list)
	pair_kf = KFold(n_fold, shuffle=True, random_state=42)
	pair_train_clusters, pair_test_clusters = [], []
	for train_idx, test_idx in pair_kf.split(pair_list):
		pair_train_clusters.append(pair_list[train_idx])
		pair_test_clusters.append(pair_list[test_idx])
	
	return pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict


def load_data(measure, setting, clu_thre, n_fold):
	data_pack = []
	pid_list = []
	cid_list = []
	with open(f"./out3_last_seq_data_{measure}.tsv", newline="") as tsvfile:
		reader = csv.reader(tsvfile, delimiter='\t')
		next(reader)
		for row in reader:
			data_pack.append(row)
			pid_list.append(row[5]) # 蛋白质序列列表（因为有的蛋白质id不同，但是蛋白质序列相同，为了和前面的统一，只可以使用蛋白质序列列表）
			cid_list.append(row[3])
	n_sample = len(cid_list)
	
	# train-test split
	train_idx_list, valid_idx_list, test_idx_list = [], [], []
	print('setting:', setting)
	if setting == 'imputation':
		pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
		= split_train_test_clusters(measure, clu_thre, n_fold)
		for fold in range(n_fold):
			pair_train_valid, pair_test = pair_train_clusters[fold], pair_test_clusters[fold]
			pair_valid = np.random.choice(pair_train_valid, int(len(pair_train_valid)*0.125), replace=False)
			pair_train = set(pair_train_valid)-set(pair_valid)
			pair_valid = set(pair_valid)
			pair_test = set(pair_test)
			train_idx, valid_idx, test_idx = [], [], []
			for ele in range(n_sample):
				if 'c'+str(C_cluster_dict[cid_list[ele]])+'p'+str(P_cluster_dict[pid_list[ele]]) in pair_train:
					train_idx.append(ele)
				elif 'c'+str(C_cluster_dict[cid_list[ele]])+'p'+str(P_cluster_dict[pid_list[ele]]) in pair_valid:
					valid_idx.append(ele)
				elif 'c'+str(C_cluster_dict[cid_list[ele]])+'p'+str(P_cluster_dict[pid_list[ele]]) in pair_test:
					test_idx.append(ele)
				else:
					print('error')
			train_idx_list.append(train_idx)
			valid_idx_list.append(valid_idx)
			test_idx_list.append(test_idx)
			# print('fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))
			
	elif setting == 'new_protein':
		pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
		= split_train_test_clusters(measure, clu_thre, n_fold)
		for fold in range(n_fold):
			p_train_valid, p_test = p_train_clusters[fold], p_test_clusters[fold]
			p_valid = np.random.choice(p_train_valid, int(len(p_train_valid)*0.125), replace=False)
			p_train = set(p_train_valid)-set(p_valid)
			train_idx, valid_idx, test_idx = [], [], []
			for ele in range(n_sample): 
				if P_cluster_dict[pid_list[ele]] in p_train:
					train_idx.append(ele)
				elif P_cluster_dict[pid_list[ele]] in p_valid:
					valid_idx.append(ele)
				elif P_cluster_dict[pid_list[ele]] in p_test:
					test_idx.append(ele)
				else:
					print('error')
			train_idx_list.append(train_idx)
			valid_idx_list.append(valid_idx)
			test_idx_list.append(test_idx)
			# print('fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))
			
	elif setting == 'new_compound':
		pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
		= split_train_test_clusters(measure, clu_thre, n_fold)
		for fold in range(n_fold):
			c_train_valid, c_test = c_train_clusters[fold], c_test_clusters[fold]
			c_valid = np.random.choice(c_train_valid, int(len(c_train_valid)*0.125), replace=False)
			c_train = set(c_train_valid)-set(c_valid)
			train_idx, valid_idx, test_idx = [], [], []
			for ele in range(n_sample):
				if C_cluster_dict[cid_list[ele]] in c_train:
					train_idx.append(ele)
				elif C_cluster_dict[cid_list[ele]] in c_valid:
					valid_idx.append(ele)
				elif C_cluster_dict[cid_list[ele]] in c_test:
					test_idx.append(ele)
				else:
					print('error')
			train_idx_list.append(train_idx)
			valid_idx_list.append(valid_idx)
			test_idx_list.append(test_idx)
			# print('fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))
	
	elif setting == 'new_new':
		assert n_fold ** 0.5 == int(n_fold ** 0.5)
		pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
		= split_train_test_clusters(measure, clu_thre, int(n_fold ** 0.5))
		
		for fold_x in range(int(n_fold ** 0.5)):
			for fold_y in range(int(n_fold ** 0.5)):
				c_train_valid, p_train_valid = c_train_clusters[fold_x], p_train_clusters[fold_y]
				c_test, p_test = c_test_clusters[fold_x], p_test_clusters[fold_y]
				c_valid = np.random.choice(list(c_train_valid), int(len(c_train_valid)/3), replace=False)
				c_train = set(c_train_valid)-set(c_valid)
				p_valid = np.random.choice(list(p_train_valid), int(len(p_train_valid)/3), replace=False)
				p_train = set(p_train_valid)-set(p_valid)
				
				train_idx, valid_idx, test_idx = [], [], []
				for ele in range(n_sample):
					if C_cluster_dict[cid_list[ele]] in c_train and P_cluster_dict[pid_list[ele]] in p_train:
						train_idx.append(ele)
					elif C_cluster_dict[cid_list[ele]] in c_valid and P_cluster_dict[pid_list[ele]] in p_valid:
						valid_idx.append(ele)
					elif C_cluster_dict[cid_list[ele]] in c_test and P_cluster_dict[pid_list[ele]] in p_test:
						test_idx.append(ele)
				train_idx_list.append(train_idx)
				valid_idx_list.append(valid_idx)
				test_idx_list.append(test_idx)
				# print('fold', fold_x*int(n_fold ** 0.5)+fold_y, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))
	return data_pack, train_idx_list, valid_idx_list, test_idx_list



