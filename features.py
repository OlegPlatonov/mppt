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

one_hop_adj_dict = {
    5: 0, 6: 1, 7: 2, 8: 3, 15: 4, 16: 5, 14: 6, 34: 7, 13: 8, 4: 9, 33: 10, 32: 11, 31: 12, 'other': 13
}

one_hop_adj_with_bonds_dict = {}
i = 0
for bond_type in range(4):
    for bond_stereo in range(3):
        for conjugated in range(2):
            for atom_id in one_hop_adj_dict:
                one_hop_adj_with_bonds_dict[(bond_type, bond_stereo, conjugated, atom_id)] = i
                i += 1
