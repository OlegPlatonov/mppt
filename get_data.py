import os
import pickle
import argparse
from collections import Counter, deque
from tqdm import tqdm

from ogb.graphproppred import GraphPropPredDataset
from ogb.lsc import PCQM4Mv2Dataset
from ogb.utils import smiles2graph

from mappings import (num_targets_dict, feature_name_to_feature_dict, feature_id_to_feature_name, feature_dims,
                      path_len_to_id)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='pcqm4mv2',
                        choices=['pcqm4mv2', 'ogbg-molpcba', 'ogbg-molhiv', 'ogbg-moltox21', 'ogbg-molbace',
                                 'ogbg-molbbbp', 'ogbg-molclintox', 'ogbg-molmuv', 'ogbg-molsider', 'ogbg-moltoxcast',
                                 'ogbg-molesol', 'ogbg-molfreesolv', 'ogbg-mollipo'])
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--min_feature_count', type=int, default=1000)
    parser.add_argument('--features_dict_dataset', type=str, default=None,
                        choices=['pcqm4mv2', 'ogbg-molpcba', 'ogbg-molhiv', 'ogbg-moltox21', 'ogbg-molbace',
                                 'ogbg-molbbbp', 'ogbg-molclintox', 'ogbg-molmuv', 'ogbg-molsider', 'ogbg-moltoxcast',
                                 'ogbg-molesol', 'ogbg-molfreesolv', 'ogbg-mollipo'],
                        help='Use features dicts from another dataset. '
                             'Only works if the features dicts for this dataset have already been precomputed.')

    args = parser.parse_args()

    return args


def get_precomputed_data(name):
    file_path = os.path.join('data', f'{name}.pickle')
    print(f'Loading precomputed data from {file_path}...')
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    return data


def get_possibly_precomputed_data(name, func, **kwargs):
    file_path = os.path.join('data', f'{name}.pickle')
    if os.path.isfile(file_path):
        data = get_precomputed_data(name)
    else:
        data = func(**kwargs)
        print(f'Saving data to {file_path}...')
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    return data


def get_orig_features(dataset, orig_feature_id):
    feature_name = feature_id_to_feature_name[orig_feature_id]
    feature_dict = feature_name_to_feature_dict[feature_name]

    features = []
    for molecule, _ in tqdm(dataset, desc=f'Computing {feature_name} features'):
        molecule_features = []
        for atom in molecule['node_feat']:
            orig_feature = atom[orig_feature_id]
            if orig_feature in feature_dict:
                atom_features = [feature_dict[orig_feature]]
            elif 'other' in feature_dict:
                atom_features = [feature_dict['other']]
            else:
                atom_features = []

            molecule_features.append(atom_features)

        features.append(molecule_features)

    return features


def get_graphs(dataset):
    atomic_num_dict = feature_name_to_feature_dict['atomic_num']

    graphs = []
    for molecule, _ in tqdm(dataset, desc='Creating graphs'):
        graph = [[] for _ in range(molecule['num_nodes'])]
        for k, (i, j) in enumerate(zip(*molecule['edge_index'])):
            bond_features = tuple(feature.item() for feature in molecule['edge_feat'][k])
            atomic_num = molecule['node_feat'][j, 0]
            atom_id = atomic_num_dict[atomic_num] if atomic_num in atomic_num_dict else atomic_num_dict['other']
            graph[i].append((bond_features, atom_id, j.item()))

        graphs.append(graph)

    return graphs


def get_one_hop_adj_counts(graphs):
    counts = []
    for molecule in tqdm(graphs, desc='Computing one hop adjacency counts'):
        molecule_counts = []
        for atom in molecule:
            atom_counts = Counter()
            for bond_features, atom_id, _ in atom:
                atom_counts[(bond_features, atom_id)] += 1

            molecule_counts.append(atom_counts)

        counts.append(molecule_counts)

    return counts


def get_two_hop_adj_counts(graphs):
    counts = []
    for molecule in tqdm(graphs, desc='Computing two hop adjacency counts'):
        molecule_counts = []
        for i, atom in enumerate(molecule):
            atom_counts = Counter()
            for one_hop_bond_features, one_hop_atom_id, one_hop_neighbor_id in atom:
                for two_hop_bond_features, two_hop_atom_id, two_hop_neighbor_id in molecule[one_hop_neighbor_id]:
                    if two_hop_neighbor_id != i:
                        atom_counts[(one_hop_bond_features, one_hop_atom_id,
                                     two_hop_bond_features, two_hop_atom_id)] += 1

            molecule_counts.append(atom_counts)

        counts.append(molecule_counts)

    return counts


def get_three_hop_adj_counts(graphs):
    counts = []
    for molecule in tqdm(graphs, desc='Computing three hop adjacency counts'):
        molecule_counts = []
        for i, atom in enumerate(molecule):
            atom_counts = Counter()
            for one_hop_bond_features, one_hop_atom_id, one_hop_neighbor_id in atom:
                for two_hop_bond_features, two_hop_atom_id, two_hop_neighbor_id in molecule[one_hop_neighbor_id]:
                    if two_hop_neighbor_id != i:
                        for (three_hop_bond_features, three_hop_atom_id,
                             three_hop_neighbor_id) in molecule[two_hop_neighbor_id]:
                            if three_hop_neighbor_id != i and three_hop_neighbor_id != one_hop_neighbor_id:
                                atom_counts[(one_hop_bond_features, one_hop_atom_id,
                                             two_hop_bond_features, two_hop_atom_id,
                                             three_hop_bond_features, three_hop_atom_id)] += 1

            molecule_counts.append(atom_counts)

        counts.append(molecule_counts)

    return counts


def get_total_feature_counts(splits, counts):
    total_counts = Counter()
    for idx in tqdm(splits['train'], desc='Computing total feature counts'):
        molecule = counts[idx]
        for atom in molecule:
            for key, count in atom.items():
                for i in range(1, count + 1):
                    total_counts[(*key, i)] += 1

    return total_counts


def get_features_dict(total_counts, min_feature_count):
    features_dict = {}
    i = 0
    for key, count in sorted(total_counts.items(), key=lambda x: x[1], reverse=True):
        if count < min_feature_count:
            break

        features_dict[key] = i
        i += 1

    return features_dict


def get_adj_features(counts, features_dict):
    features = []
    for molecule in tqdm(counts, desc='Computing features'):
        molecule_features = []
        for atom in molecule:
            atom_features = []
            for key, count in atom.items():
                for i in range(1, count + 1):
                    feature = (*key, i)
                    if feature in features_dict:
                        atom_features.append(features_dict[feature])

            molecule_features.append(atom_features)

        features.append(molecule_features)

    return features


def merge_features(dataset, all_features):
    merged_features = []
    for molecule, _ in dataset:
        molecule_features = [[] for _ in range(molecule['num_nodes'])]
        merged_features.append(molecule_features)

    shift = 0
    for feature_name, features in tqdm(all_features.items(), desc='Merging features'):
        for i in range(len(features)):
            for j in range(len(features[i])):
                for feature in features[i][j]:
                    merged_features[i][j].append(feature + shift)

        feature_dim = feature_dims[feature_name]
        shift += feature_dim

    return merged_features


def add_virtual_nodes(features):
    for i in range(len(features)):
        features[i] = [[0]] + features[i]

    return features


def get_features_string(features):
    molecule_strings = []
    for atom in features:
        atom_strings = [str(num) for num in atom]
        atom_strings_joined = ','.join(atom_strings)
        molecule_strings.append(atom_strings_joined)

    molecule_strings_joined = ';'.join(molecule_strings)

    return molecule_strings_joined


def get_paths_string(paths):
    path_lengths = [[None for _ in range(len(paths) + 1)] for _ in range(len(paths) + 1)]

    for i in range(len(paths) + 1):
        path_lengths[0][i] = 'virtual_to_any'

    for i in range(len(paths) + 1):
        path_lengths[i][0] = 'any_to_virtual'

    path_lengths[0][0] = 'virtual_to_virtual'

    for i in range(len(paths)):
        for j in range(len(paths)):
            path = paths[i][j]
            if path is None:
                length = None
            else:
                length = (len(path) + 1) // 2
                if length > 7:
                    length = 'far'

            path_lengths[i + 1][j + 1] = length

    path_ids = [[None for _ in range(len(paths) + 1)] for _ in range(len(paths) + 1)]
    for i in range(len(path_ids)):
        for j in range(len(path_ids)):
            path_ids[i][j] = path_len_to_id[path_lengths[i][j]]

    molecule_strings = []
    for atom in path_ids:
        atom_strings = [str(num) for num in atom]
        atom_strings_joined = ','.join(atom_strings)
        molecule_strings.append(atom_strings_joined)

    molecule_strings_joined = ';'.join(molecule_strings)

    return molecule_strings_joined


def target_to_str(target):
    if isinstance(target, float):
        target_string = str(target)
    else:
        target_string = ','.join(str(num) for num in target)

    return target_string


def get_strings(dataset, features, paths):
    strings = []
    for i in tqdm(range(len(features)), desc='Converting features to strings'):
        features_string = get_features_string(features[i])
        paths_string = get_paths_string(paths[i])
        _, target = dataset[i]
        target_string = target_to_str(target)
        full_string = '|'.join([features_string, paths_string, target_string])
        strings.append(full_string)

    return strings


def get_shortest_paths(graphs):
    paths = []
    for graph in tqdm(graphs, desc='Computing shortest paths'):
        cur_paths = [[None for _ in graph] for _ in graph]
        for source_node_id in range(len(graph)):
            cur_paths[source_node_id][source_node_id] = ()
            queue = deque([source_node_id])
            while queue:
                node_id = queue.popleft()
                for bond_features, atom_id, neighbor_id in graph[node_id]:
                    if cur_paths[source_node_id][neighbor_id] is None:
                        cur_paths[source_node_id][neighbor_id] = (cur_paths[source_node_id][node_id] +
                                                                  (bond_features, atom_id))
                        queue.append(neighbor_id)

        for i in range(len(cur_paths)):
            for j in range(len(cur_paths)):
                if cur_paths[i][j] is not None:
                    cur_paths[i][j] = cur_paths[i][j][:-1]

        paths.append(cur_paths)

    return paths


def main():
    args = get_args()

    if args.dataset == 'pcqm4mv2':
        dataset = PCQM4Mv2Dataset(root='data', smiles2graph=smiles2graph)
    else:
        dataset = GraphPropPredDataset(name=args.dataset, root='data')

    splits = dataset.get_idx_split()

    all_features = {}

    # get original OGB features
    for orig_feature_id, feature_name in feature_id_to_feature_name.items():
        features = get_possibly_precomputed_data(name=f'{args.dataset}_{feature_name}_features',
                                                 func=get_orig_features,
                                                 dataset=dataset,
                                                 orig_feature_id=orig_feature_id)
        all_features[feature_name] = features

    graphs = get_possibly_precomputed_data(name=f'{args.dataset}_graphs', func=get_graphs, dataset=dataset)

    # get one hop adjacency features
    one_hop_adj_counts = get_possibly_precomputed_data(name=f'{args.dataset}_one_hop_adj_counts',
                                                       func=get_one_hop_adj_counts,
                                                       graphs=graphs)

    total_one_hop_adj_counts = get_possibly_precomputed_data(name=f'{args.dataset}_total_one_hop_adj_counts',
                                                             func=get_total_feature_counts,
                                                             splits=splits,
                                                             counts=one_hop_adj_counts)

    if args.features_dict_dataset is not None:
        name = f'{args.features_dict_dataset}_one_hop_adj_features_dict'
        one_hop_adj_features_dict = get_precomputed_data(name=name)
    else:
        one_hop_adj_features_dict = get_possibly_precomputed_data(name=f'{args.dataset}_one_hop_adj_features_dict',
                                                                  func=get_features_dict,
                                                                  total_counts=total_one_hop_adj_counts,
                                                                  min_feature_count=args.min_feature_count)

    name = f'{args.dataset}_one_hop_adj_features' if args.features_dict_dataset is None else \
        f'{args.dataset}_for_{args.features_dict_dataset}_one_hop_adj_features'
    one_hop_adj_features = get_possibly_precomputed_data(name=name,
                                                         func=get_adj_features,
                                                         counts=one_hop_adj_counts,
                                                         features_dict=one_hop_adj_features_dict)

    all_features['one_hop_adj'] = one_hop_adj_features
    feature_dims['one_hop_adj'] = len(one_hop_adj_features_dict)

    # get two hop adjacency features
    two_hop_adj_counts = get_possibly_precomputed_data(name=f'{args.dataset}_two_hop_adj_counts',
                                                       func=get_two_hop_adj_counts,
                                                       graphs=graphs)

    total_two_hop_adj_counts = get_possibly_precomputed_data(name=f'{args.dataset}_total_two_hop_adj_counts',
                                                             func=get_total_feature_counts,
                                                             splits=splits,
                                                             counts=two_hop_adj_counts)

    if args.features_dict_dataset is not None:
        name = f'{args.features_dict_dataset}_two_hop_adj_features_dict'
        two_hop_adj_features_dict = get_precomputed_data(name=name)
    else:
        two_hop_adj_features_dict = get_possibly_precomputed_data(name=f'{args.dataset}_two_hop_adj_features_dict',
                                                                  func=get_features_dict,
                                                                  total_counts=total_two_hop_adj_counts,
                                                                  min_feature_count=args.min_feature_count)

    name = f'{args.dataset}_two_hop_adj_features' if args.features_dict_dataset is None else \
        f'{args.dataset}_for_{args.features_dict_dataset}_two_hop_adj_features'
    two_hop_adj_features = get_possibly_precomputed_data(name=name,
                                                         func=get_adj_features,
                                                         counts=two_hop_adj_counts,
                                                         features_dict=two_hop_adj_features_dict)

    all_features['two_hop_adj'] = two_hop_adj_features
    feature_dims['two_hop_adj'] = len(two_hop_adj_features_dict)

    # get three hop adjacency features
    three_hop_adj_counts = get_possibly_precomputed_data(name=f'{args.dataset}_three_hop_adj_counts',
                                                         func=get_three_hop_adj_counts,
                                                         graphs=graphs)

    total_three_hop_adj_counts = get_possibly_precomputed_data(name=f'{args.dataset}_total_three_hop_adj_counts',
                                                               func=get_total_feature_counts,
                                                               splits=splits,
                                                               counts=three_hop_adj_counts)

    if args.features_dict_dataset is not None:
        name = f'{args.features_dict_dataset}_three_hop_adj_features_dict'
        three_hop_adj_features_dict = get_precomputed_data(name=name)
    else:
        three_hop_adj_features_dict = get_possibly_precomputed_data(name=f'{args.dataset}_three_hop_adj_features_dict',
                                                                    func=get_features_dict,
                                                                    total_counts=total_three_hop_adj_counts,
                                                                    min_feature_count=args.min_feature_count)

    name = f'{args.dataset}_three_hop_adj_features' if args.features_dict_dataset is None else \
        f'{args.dataset}_for_{args.features_dict_dataset}_three_hop_adj_features'
    three_hop_adj_features = get_possibly_precomputed_data(name=name,
                                                           func=get_adj_features,
                                                           counts=three_hop_adj_counts,
                                                           features_dict=three_hop_adj_features_dict)

    all_features['three_hop_adj'] = three_hop_adj_features
    feature_dims['three_hop_adj'] = len(three_hop_adj_features_dict)

    features_merged = merge_features(dataset=dataset, all_features=all_features)

    features_merged = add_virtual_nodes(features_merged)

    paths = get_possibly_precomputed_data(name=f'{args.dataset}_shortest_paths', func=get_shortest_paths, graphs=graphs)

    strings = get_strings(dataset=dataset, features=features_merged, paths=paths)

    num_features = sum(feature_dims[feature_name] for feature_name in all_features)
    num_path_types = len(path_len_to_id)
    num_targets = num_targets_dict[args.dataset]

    name = f'{args.dataset}_{args.output_name}' if args.features_dict_dataset is None else \
        f'{args.dataset}_for_{args.features_dict_dataset}_{args.output_name}'

    file_path_train = os.path.join('data', name + '_train.csv')
    print(f'Saving train data to {file_path_train}...')
    with open(file_path_train, 'w') as file:
        file.write(f'{num_features},{num_path_types},{num_targets}\n')
        for i in splits['train']:
            file.write(strings[i] + '\n')

    file_path_val = os.path.join('data', name + '_val.csv')
    print(f'Saving val data to {file_path_val}...')
    with open(file_path_val, 'w') as file:
        file.write(f'{num_features},{num_path_types},{num_targets}\n')
        for i in splits['valid']:
            file.write(strings[i] + '\n')

    if args.dataset != 'pcqm4mv2':
        file_path_test = os.path.join('data', name + '_test.csv')
        print(f'Saving test data to {file_path_test}...')
        with open(file_path_test, 'w') as file:
            file.write(f'{num_features},{num_path_types},{num_targets}\n')
            for i in splits['test']:
                file.write(strings[i] + '\n')


if __name__ == '__main__':
    main()
