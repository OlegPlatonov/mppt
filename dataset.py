import torch
from torch.utils.data import Dataset


class MoleculeDataset(Dataset):
    def __init__(self, name, split):
        filename = f'data/{name}_{split}.csv'
        with open(filename) as file:
            first_line = next(file).strip()
            lines = [line.strip() for line in file]

        input_dim, num_rels, num_targets = first_line.split(',')

        self.input_dim = int(input_dim)
        self.num_rels = int(num_rels) + 1   # +1 for padding
        self.num_targets = int(num_targets)
        self.data = lines

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        string = self.data[idx]
        all_atoms_string, all_rels_string, target_string = string.split('|')
        if self.num_targets == 1:
            target = float(target_string)
        else:
            target = [float(num) for num in target_string.split(',')]

        atom_strings = all_atoms_string.split(';')
        num_atoms = len(atom_strings)

        features = torch.zeros(num_atoms, self.input_dim)
        for atom_id, atom_string in enumerate(atom_strings):
            for feature_id in [int(num) for num in atom_string.split(',')]:
                features[atom_id, feature_id] = 1

        rel_strings = all_rels_string.split(';')
        rels = torch.zeros(num_atoms, num_atoms, dtype=int)
        for i, rel_string in enumerate(rel_strings):
            for j, rel in enumerate(rel_string.split(',')):
                rels[i, j] = int(rel)

        return features, rels, target

    @staticmethod
    def pad_molecules(molecules):
        dim = len(molecules[0][0])
        padding_vector = torch.zeros(dim)

        max_len = max(len(molecule) for molecule in molecules)
        attn_mask = torch.zeros(len(molecules), max_len)
        for i in range(len(molecules)):
            cur_len = len(molecules[i])
            attn_mask[i, :cur_len] = 1
            cur_padding_len = max_len - cur_len
            cur_padding_matrix = torch.tile(padding_vector, (cur_padding_len, 1))
            molecules[i] = torch.cat([molecules[i], cur_padding_matrix], axis=0)

        molecules = torch.stack(molecules)

        return molecules, attn_mask

    @staticmethod
    def pad_relations(rels, pad_value):
        max_len = max(len(molecule_rels) for molecule_rels in rels)
        for i, molecule_rels in enumerate(rels):
            padded_molecule_rels = []
            cur_padding = pad_value * torch.ones(max_len - len(molecule_rels), dtype=int)
            for atom_rels in molecule_rels:
                padded_atom_rels = torch.cat([atom_rels, cur_padding], axis=0)
                padded_molecule_rels.append(padded_atom_rels)

            lower_padding = pad_value * torch.ones(max_len, dtype=int)
            for _ in range(max_len - len(molecule_rels)):
                padded_molecule_rels.append(lower_padding)

            rels[i] = torch.vstack(padded_molecule_rels)

        rels = torch.stack(rels, dim=0)

        return rels

    @staticmethod
    def transform_attention_mask(attn_mask):
        attn_mask = 1 - attn_mask
        attn_mask[attn_mask == 1] = - torch.inf
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

        return attn_mask

    def collate_fn(self, batch):
        molecules = []
        rels = []
        targets = []
        for molecule, molecule_rels, target in batch:
            molecules.append(molecule)
            rels.append(molecule_rels)
            targets.append(target)

        molecules, attn_mask = self.pad_molecules(molecules)
        rels = self.pad_relations(rels, pad_value=self.num_rels - 1)
        attn_mask = self.transform_attention_mask(attn_mask)
        targets = torch.tensor(targets)

        return molecules, rels, attn_mask, targets
