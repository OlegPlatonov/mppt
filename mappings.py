metric_names = {
    'ogbg-molpcba': 'AP',
    'ogbg-molmuv': 'AP',
    'ogbg-molhiv': 'ROC AUC',
    'ogbg-moltoxcast': 'ROC AUC',
    'ogbg-moltox21': 'ROC AUC',
    'ogbg-molbbbp': 'ROC AUC',
    'ogbg-molbace': 'ROC AUC',
    'ogbg-molclintox': 'ROC AUC',
    'ogbg-molsider': 'ROC AUC',
    'ogbg-mollipo': 'RMSE',
    'ogbg-molesol': 'RMSE',
    'ogbg-molfreesolv': 'RMSE',
    'pcqm4mv2': 'MAE'
}

ogb_metric_names = {
    'ogbg-molpcba': 'ap',
    'ogbg-molmuv': 'ap',
    'ogbg-molhiv': 'rocauc',
    'ogbg-moltoxcast': 'rocauc',
    'ogbg-moltox21': 'rocauc',
    'ogbg-molbbbp': 'rocauc',
    'ogbg-molbace': 'rocauc',
    'ogbg-molclintox': 'rocauc',
    'ogbg-molsider': 'rocauc',
    'ogbg-mollipo': 'rmse',
    'ogbg-molesol': 'rmse',
    'ogbg-molfreesolv': 'rmse'
}

maximize_metric = {
    'ogbg-molpcba': True,
    'ogbg-molmuv': True,
    'ogbg-molhiv': True,
    'ogbg-moltoxcast': True,
    'ogbg-moltox21': True,
    'ogbg-molbbbp': True,
    'ogbg-molbace': True,
    'ogbg-molclintox': True,
    'ogbg-molsider': True,
    'ogbg-mollipo': False,
    'ogbg-molesol': False,
    'ogbg-molfreesolv': False,
    'pcqm4mv2': False
}

num_targets_dict = {
    'ogbg-molpcba': 128,
    'ogbg-molmuv': 17,
    'ogbg-molhiv': 1,
    'ogbg-moltoxcast': 617,
    'ogbg-moltox21': 12,
    'ogbg-molbbbp': 1,
    'ogbg-molbace': 1,
    'ogbg-molclintox': 2,
    'ogbg-molsider': 27,
    'ogbg-mollipo': 1,
    'ogbg-molesol': 1,
    'ogbg-molfreesolv': 1,
    'pcqm4mv2': 1
}


atomic_num_dict = {
    'virtual': 0, 5: 1, 6: 2, 7: 3, 8: 4, 15: 5, 16: 6, 14: 7, 34: 8, 13: 9, 4: 10, 33: 11, 32: 12, 31: 13, 'other': 14
}

chirality_dict = {
    0: 0, 1: 1, 2: 2
}

degree_dict = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6
}

formal_charge_dict = {
    4: 0, 5: 1, 6: 2
}

numH_dict = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4
}

num_radical_e_dict = {
    0: 0, 1: 1, 2: 2, 3: 3
}

hybridization_dict = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5
}

aromatic_dict = {
    1: 0
}

in_ring_dict = {
    1: 0
}

bond_type_dict = {
    0: 0, 1: 1, 2: 2, 3: 3
}

bond_stereo_dict = {
    0: 0, 1: 1, 2: 2
}

conjugated_dict = {
    1: 0
}


feature_name_to_feature_dict = {
    'atomic_num': atomic_num_dict,
    'chirality': chirality_dict,
    'degree': degree_dict,
    'formal_charge': formal_charge_dict,
    'numH': numH_dict,
    'num_radical_e': num_radical_e_dict,
    'hybridization': hybridization_dict,
    'aromatic': aromatic_dict,
    'in_ring': in_ring_dict
}

feature_id_to_feature_name = {
    0: 'atomic_num',
    1: 'chirality',
    2: 'degree',
    3: 'formal_charge',
    4: 'numH',
    5: 'num_radical_e',
    6: 'hybridization',
    7: 'aromatic',
    8: 'in_ring'
}

feature_dims = {
    'atomic_num': len(atomic_num_dict),
    'chirality': len(chirality_dict),
    'degree': len(degree_dict),
    'formal_charge': len(formal_charge_dict),
    'numH': len(numH_dict),
    'num_radical_e': len(num_radical_e_dict),
    'hybridization': len(hybridization_dict),
    'aromatic': len(aromatic_dict),
    'in_ring': len(in_ring_dict),
    'bond_type': len(bond_type_dict),
    'bond_stereo': len(bond_stereo_dict),
    'conjugated': len(conjugated_dict),
}


path_len_to_id = {
    'virtual_to_virtual': 0,
    'virtual_to_any': 1,
    'any_to_virtual': 2,
    'far': 11,
    None: 12
}

for length in range(8):
    path_len_to_id[length] = length + 3
