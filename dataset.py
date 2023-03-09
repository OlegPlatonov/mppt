import torch
from torch.utils.data import Dataset


class MoleculeDataset(Dataset):
    def __init__(self, name, split):
        filename = f'data/{name}_{split}.csv'
        with open(filename) as file:
            first_line = next(file).strip()
            lines = [line.strip() for line in file]

        input_dim, num_targets = first_line.split(',')

        self.input_dim = int(input_dim)
        self.num_targets = int(num_targets)
        self.data = lines

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        string = self.data[idx]
        string, target = string.split('|')
        target = float(target)

        object_strings = string.split(';')
        num_objects = len(object_strings)

        features = torch.zeros(num_objects, self.input_dim)
        for object_id, object_string in enumerate(object_strings):
            for feature_id in [int(num) for num in object_string.split(',')]:
                features[object_id, feature_id] = 1

        return features, target


def pad(molecules):
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


def transform_attention_mask(attn_mask):
    attn_mask = 1 - attn_mask
    attn_mask[attn_mask == 1] = - torch.inf
    attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

    return attn_mask


def collate_fn(batch):
    molecules = []
    targets = []
    for molecule, target in batch:
        molecules.append(molecule)
        targets.append(target)

    molecules, attn_mask = pad(molecules)
    attn_mask = transform_attention_mask(attn_mask)
    targets = torch.tensor(targets)

    return molecules, attn_mask, targets
