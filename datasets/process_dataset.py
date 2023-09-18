import os
import random
import time
import math
import pickle
from functools import partial
from multiprocessing import Pool
import bisect
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
import torch
import pandas as pd
# from rdkit import Chem
from tqdm import tqdm
from scipy.sparse import lil_matrix, vstack

from utils import mkdir
from datasets.preprocess import (
    mapping, graphs_to_min_dfscodes,
    min_dfscodes_to_tensors, random_walk_with_restart_sampling
)


def check_graph_size(
    graph, min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):

    if min_num_nodes and graph.number_of_nodes() < min_num_nodes:
        return False
    if max_num_nodes and graph.number_of_nodes() > max_num_nodes:
        return False

    if min_num_edges and graph.number_of_edges() < min_num_edges:
        return False
    if max_num_edges and graph.number_of_edges() > max_num_edges:
        return False

    return True


def produce_graphs_from_raw_format(
    inputfile, output_path, num_graphs=None, min_num_nodes=None,
    max_num_nodes=None, min_num_edges=None, max_num_edges=None
):
    """
    :param inputfile: Path to file containing graphs in raw format
    :param output_path: Path to store networkx graphs
    :param num_graphs: Upper bound on number of graphs to be taken
    :param min_num_nodes: Lower bound on number of nodes in graphs if provided
    :param max_num_nodes: Upper bound on number of nodes in graphs if provided
    :param min_num_edges: Lower bound on number of edges in graphs if provided
    :param max_num_edges: Upper bound on number of edges in graphs if provided
    :return: number of graphs produced
    """

    lines = []
    with open(inputfile, 'r') as fr:
        for line in fr:
            line = line.strip().split()
            lines.append(line)

    index = 0
    count = 0
    graphs_ids = set()
    while index < len(lines):
        if lines[index][0][1:] not in graphs_ids:
            graph_id = lines[index][0][1:]
            G = nx.Graph(id=graph_id)

            index += 1
            vert = int(lines[index][0])
            index += 1
            for i in range(vert):
                G.add_node(i, label=lines[index][0])
                index += 1

            edges = int(lines[index][0])
            index += 1
            for i in range(edges):
                G.add_edge(int(lines[index][0]), int(
                    lines[index][1]), label=lines[index][2])
                index += 1

            index += 1

            if not check_graph_size(
                G, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
            ):
                continue

            if nx.is_connected(G):
                with open(os.path.join(
                        output_path, 'graph{}.dat'.format(count)), 'wb') as f:
                    pickle.dump(G, f)

                graphs_ids.add(graph_id)
                count += 1

                if num_graphs and count >= num_graphs:
                    break

        else:
            vert = int(lines[index + 1][0])
            edges = int(lines[index + 2 + vert][0])
            index += vert + edges + 4

    return count


# For Enzymes dataset
def produce_graphs_from_graphrnn_format(
    input_path, dataset_name, output_path, num_graphs=None,
    node_invariants=[], min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):
    node_attributes = False
    graph_labels = False

    G = nx.Graph()
    # load data
    path = input_path
    data_adj = np.loadtxt(os.path.join(path, dataset_name + '_A.txt'),
                          delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(
            os.path.join(path, dataset_name + '_node_attributes.txt'),
            delimiter=',')

    data_node_label = np.loadtxt(
        os.path.join(path, dataset_name + '_node_labels.txt'),
        delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(
        os.path.join(path, dataset_name + '_graph_indicator.txt'),
        delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(
            os.path.join(path, dataset_name + '_graph_labels.txt'),
            delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)

    # add node labels
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
        G.add_node(i + 1, label=str(data_node_label[i]))

    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1

    count = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['id'] = data_graph_labels[i]

        if not check_graph_size(
            G_sub, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
        ):
            continue

        if nx.is_connected(G_sub):
            G_sub = nx.convert_node_labels_to_integers(G_sub)
            G_sub.remove_edges_from(nx.selfloop_edges(G_sub))

            if 'CC' in node_invariants:
                clustering_coeff = nx.clustering(G_sub)
                cc_bins = [0, 0.2, 0.4, 0.6, 0.8]

            for node in G_sub.nodes():
                node_label = str(G_sub.nodes[node]['label'])

                if 'Degree' in node_invariants:
                    node_label += '-' + str(G_sub.degree[node])

                if 'CC' in node_invariants:
                    node_label += '-' + str(
                        bisect.bisect(cc_bins, clustering_coeff[node]))

                G_sub.nodes[node]['label'] = node_label

            nx.set_edge_attributes(G_sub, 'DEFAULT_LABEL', 'label')

            with open(os.path.join(
                    output_path, 'graph{}.dat'.format(count)), 'wb') as f:
                pickle.dump(G_sub, f)

            count += 1

            if num_graphs and count >= num_graphs:
                break

    return count

def adj_to_graph(adj, is_cuda=False):
    # if is_cuda:
    adj = adj.detach().cpu().numpy()
    G = nx.from_numpy_matrix(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() < 1:
        G.add_node(1)
    return G

def sample_subgraphs(
    idx, G, output_path, iterations, num_factor, min_num_nodes=None,
    max_num_nodes=None, min_num_edges=None, max_num_edges=None
):
    count = 0
    deg = G.degree[idx]
    for _ in range(num_factor * int(math.sqrt(deg))):
        G_rw = random_walk_with_restart_sampling(
            G, idx, iterations=iterations, max_nodes=max_num_nodes,
            max_edges=max_num_edges)
        G_rw = nx.convert_node_labels_to_integers(G_rw)
        G_rw.remove_edges_from(nx.selfloop_edges(G_rw))

        if not check_graph_size(
            G_rw, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
        ):
            continue

        if nx.is_connected(G_rw):
            with open(os.path.join(
                    output_path,
                    'graph{}-{}.dat'.format(idx, count)), 'wb') as f:
                pickle.dump(G_rw, f)
                count += 1


def produce_random_walk_sampled_graphs(
    input_path, dataset_name, output_path, iterations, num_factor,
    num_graphs=None, min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):
    print('Producing random_walk graphs - num_factor - {}'.format(num_factor))
    G = nx.Graph()

    d = {}
    count = 0
    with open(os.path.join(input_path, dataset_name + '.content'), 'r') as f:
        for line in f.readlines():
            spp = line.strip().split('\t')
            G.add_node(count, label=spp[-1])
            d[spp[0]] = count
            count += 1

    count = 0
    with open(os.path.join(input_path, dataset_name + '.cites'), 'r') as f:
        for line in f.readlines():
            spp = line.strip().split('\t')
            if spp[0] in d and spp[1] in d:
                G.add_edge(d[spp[0]], d[spp[1]], label='DEFAULT_LABEL')
            else:
                count += 1

    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G)

    with Pool(processes=48) as pool:
        for _ in tqdm(pool.imap_unordered(partial(
                sample_subgraphs, G=G, output_path=output_path,
                iterations=iterations, num_factor=num_factor,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges),
                list(range(G.number_of_nodes())))):
            pass

    filenames = []
    for name in os.listdir(output_path):
        if name.endswith('.dat'):
            filenames.append(name)

    random.shuffle(filenames)

    if not num_graphs:
        num_graphs = len(filenames)

    count = 0
    for i, name in enumerate(filenames[:num_graphs]):
        os.rename(
            os.path.join(output_path, name),
            os.path.join(output_path, 'graph{}.dat'.format(i))
        )
        count += 1

    for name in filenames[num_graphs:]:
        os.remove(os.path.join(output_path, name))

    return count

def add_node_label(graph):
    G = nx.convert_node_labels_to_integers(graph)
    for node in G.nodes():
        node_label = str(G.degree[node])
        G.nodes[node]['label'] = node_label
        
    nx.set_edge_attributes(G, 'DEFAULT_LABEL', 'label')
    return G
    
def produce_graphs_from_gdss_format(input_path, output_path):
    if input_path.endswith('.pkl'):
        with open(input_path, 'rb') as f:
            graphs = pickle.load(f)
    else:
        adjs, _, _, _, _, _, _, _ = torch.load(f'{input_path}')
        graphs = [adj_to_graph(adj) for adj in adjs]
    
    graphs = process_graphs(graphs, output_path)

    return len(graphs)

def process_graphs(graphs, output_path):
    graphs = [max([G.subgraph(c) for c in nx.connected_components(G)], key=len) for G in graphs]
    
    # add node labels (degree)    
    graphs = [add_node_label(graph) for graph in graphs]
    
    for i, graph in enumerate(graphs):
        with open(os.path.join(output_path, f'graph{i}.dat'), 'wb') as f:
            pickle.dump(graph, f)
    return graphs

def produce_graphs_lobster(output_path):
    graphs = []
    p1 = 0.7
    p2 = 0.7
    count = 0
    min_node = 10
    max_node = 100
    max_edge = 0
    mean_node = 80
    num_graphs = 100

    seed_tmp = 1234
    while count < num_graphs:
        G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
        if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
            graphs.append(G)
            if G.number_of_edges() > max_edge:
                max_edge = G.number_of_edges()
            count += 1
        seed_tmp += 1
    graphs = process_graphs(graphs, output_path)
    return len(graphs)

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def produce_graphs_ego(input_path, output_path):
    _, _, G = load_ego_data(input_path, dataset='citeseer')
    G = max([G.subgraph(c) for c in nx.connected_components(G)], key=len)
    G = nx.convert_node_labels_to_integers(G)
    graphs = []
    for i in range(G.number_of_nodes()):
        G_ego = nx.ego_graph(G, i, radius=3)
        if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
            graphs.append(G_ego)
    graphs = process_graphs(graphs, output_path)
    return len(graphs)

def load_ego_data(input_path, dataset):
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        load = pickle.load(open(f"{input_path}/ego/ind.{dataset}.{names[i]}", 'rb'), encoding='latin1')
        objects.append(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(f"{input_path}/ego/ind.{dataset}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
    tx_extended = lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range - min(test_idx_range), :] = tx
    tx = tx_extended

    features = vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    return adj, features, G

def produce_graphs_proteins(input_path, output_path):
    adjs = load_proteins_data(input_path)
    graphs = [adj_to_graph(adj.numpy()) for adj in adjs]
    graphs = process_graphs(graphs, output_path)
    return len(graphs)

def load_proteins_data(data_dir):
    
    min_num_nodes=100
    max_num_nodes=500
    
    adjs = []
    eigvals = []
    eigvecs = []
    n_nodes = []
    n_max = 0
    max_eigval = 0
    min_eigval = 0

    G = nx.Graph()
    # Load data
    path = os.path.join(data_dir, 'proteins/DD')
    data_adj = np.loadtxt(os.path.join(path, 'DD_A.txt'), delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(os.path.join(path, 'DD_graph_indicator.txt'), delimiter=',').astype(int)
    data_graph_types = np.loadtxt(os.path.join(path, 'DD_graph_labels.txt'), delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # Add edges
    G.add_edges_from(data_tuple)
    G.remove_nodes_from(list(nx.isolates(G)))

    # remove self-loop
    G.remove_edges_from(nx.selfloop_edges(G))

    # Split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1

    for i in tqdm(range(graph_num)):
        # Find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        G_sub.graph['label'] = data_graph_types[i]
        if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes() <= max_num_nodes:
            adj = torch.from_numpy(nx.adjacency_matrix(G_sub).toarray()).float()
            L = nx.normalized_laplacian_matrix(G_sub).toarray()
            L = torch.from_numpy(L).float()
            eigval, eigvec = torch.linalg.eigh(L)
            
            eigvals.append(eigval)
            eigvecs.append(eigvec)
            adjs.append(adj)
            n_nodes.append(G_sub.number_of_nodes())
            if G_sub.number_of_nodes() > n_max:
                n_max = G_sub.number_of_nodes()
            max_eigval = torch.max(eigval)
            if max_eigval > max_eigval:
                max_eigval = max_eigval
            min_eigval = torch.min(eigval)
            if min_eigval < min_eigval:
                min_eigval = min_eigval

    return adjs

def produce_graphs_point(input_path, output_path):
    graphs = load_point_data(input_path, min_num_nodes=0, max_num_nodes=10000, 
                                  node_attributes=False, graph_labels=True)
    graphs = process_graphs(graphs, output_path)
    return len(graphs)

def load_point_data(data_dir, min_num_nodes, max_num_nodes, node_attributes, graph_labels):
    print('Loading point cloud dataset')
    name = 'FIRSTMM_DB'
    G = nx.Graph()
    # load data
    path = os.path.join(data_dir, name)
    data_adj = np.loadtxt(
        os.path.join(path, f'{name}_A.txt'), delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(os.path.join(path, f'{name}_node_attributes.txt'), 
                                   delimiter=',')
    data_node_label = np.loadtxt(os.path.join(path, f'{name}_node_labels.txt'), 
                                 delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(os.path.join(path, f'{name}_graph_indicator.txt'),
                                      delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(os.path.join(path, f'{name}_graph_labels.txt'), 
                                       delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
            G.add_node(i + 1, label=data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # remove self-loop
    G.remove_edges_from(nx.selfloop_edges(G))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]

        if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes() <= max_num_nodes:
            graphs.append(G_sub)
        if G_sub.number_of_nodes() > max_nodes:
            max_nodes = G_sub.number_of_nodes()
            
    print('Loaded')
    return graphs

def produce_graphs_from_profold(input_path, output_path):
    adjs = np.load(input_path)
    adjs = adjs[:500]
    graphs = [adj_to_graph(adj) for adj in adjs]
    graphs = [max([G.subgraph(c) for c in nx.connected_components(G)], key=len) for G in graphs]
    
    # add node labels (degree)    
    graphs = [add_node_label(graph) for graph in graphs]
    
    for i, graph in enumerate(graphs):
        with open(os.path.join(output_path, f'graph{i}.dat'), 'wb') as f:
            pickle.dump(graph, f)
    return len(graphs)
    

# def smiles_to_mols(smiles):
#     return [Chem.MolFromSmiles(s) for s in tqdm(smiles, 'SMILES to molecules')]

# def canonicalize_smiles(smiles):
#     return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in tqdm(smiles, 'Canonicalizing')]

# def mols_to_nx(mols):
#     nx_graphs = []
#     for mol in tqdm(mols, 'Molecules to graph'):
#         if not mol:
#             continue
#         G = nx.Graph()

#         for atom in mol.GetAtoms():
#             G.add_node(atom.GetIdx(),
#                        label=atom.GetSymbol())
#         for bond in mol.GetBonds():
#             G.add_edge(bond.GetBeginAtomIdx(),
#                        bond.GetEndAtomIdx(),
#                        label=str(bond.GetBondTypeAsDouble()))
#         nx_graphs.append(G)
        
#     return nx_graphs

# def produce_graphs_from_molecule(input_path, output_path, data_name):
#     col_dict = {'qm9': 'SMILES1', 'zinc': 'smiles'}
#     df = pd.read_csv(input_path)
#     smiles = list(df[col_dict[data_name]])
#     smiles = [s for s in smiles if len(s)>1]
#     smiles = canonicalize_smiles(smiles)
#     mols = smiles_to_mols(smiles)
#     graphs = mols_to_nx(mols)
#     # graphs = [add_node_label_molecule(graph) for graph in graphs]
#     for i, graph in enumerate(graphs):
#         with open(os.path.join(output_path, f'graph{i}.dat'), 'wb') as f:
#             pickle.dump(graph, f)
#     return len(graphs)

# Routine to create datasets
def create_graphs(args):
    # Different datasets
    if 'Lung' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Lung/')
        input_path = base_path + 'lung.txt'
        min_num_nodes, max_num_nodes = None, 50
        min_num_edges, max_num_edges = None, None

    elif 'Breast' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Breast/')
        input_path = base_path + 'breast.txt'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'Leukemia' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Leukemia/')
        input_path = base_path + 'leukemia.txt'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'Yeast' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Yeast/')
        input_path = base_path + 'yeast.txt'
        min_num_nodes, max_num_nodes = None, 50
        min_num_edges, max_num_edges = None, None

    elif 'All' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'All/')
        input_path = base_path + 'all.txt'
        # No limit on number of nodes and edges
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'ENZYMES' in args.graph_type:
        base_path = os.path.join(args.dataset_path, 'ENZYMES/')
        # Node invariants - Options 'Degree' and 'CC'
        node_invariants = ['Degree']
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'citeseer' in args.graph_type:
        base_path = os.path.join(args.dataset_path, 'citeseer/')
        random_walk_iterations = 150  # Controls size of graph
        num_factor = 5  # Controls size of dataset

        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = 20, None

    elif 'cora' in args.graph_type:
        base_path = os.path.join(args.dataset_path, 'cora/')
        random_walk_iterations = 150  # Controls size of graph
        num_factor = 5  # Controls size of dataset

        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = 20, None
        
    elif args.graph_type in ['com_small', 'ego_small', 'grid', 'enz', 'planar', 'sbm', 
                       'ego', 'point', 'proteins', 'lobster']:
        data_name = args.graph_type
        min_num_nodes, max_num_nodes, min_num_edges, max_num_edges = None, None, None, None
        base_path = os.path.join(args.dataset_path, f'{data_name}/')
        if data_name in ['planar', 'sbm']:
            input_path = base_path + f'{data_name}.pt'
        elif data_name in ['com_small', 'ego_small', 'grid', 'enz']:
            input_path = base_path + f'{data_name}.pkl'
        elif args.graph_type in ['qm9', 'zinc']:
            input_path = base_path + f'{data_name}.csv'
        elif args.graph_type == 'profold':
            input_path = base_path + f'profold_adj.npy'
        elif args.graph_type in ['ego', 'proteins', 'point']:
            input_path = base_path
    else:
        print('Dataset - {} is not valid'.format(args.graph_type))
        exit()

    args.current_dataset_path = os.path.join(base_path, 'graphs/')
    args.min_dfscode_path = os.path.join(base_path, 'min_dfscodes/')
    min_dfscode_tensor_path = os.path.join(base_path, 'min_dfscode_tensors/')

    if args.note == 'GraphRNN' or args.note == 'DGMG':
        args.current_processed_dataset_path = args.current_dataset_path
    elif args.note == 'DFScodeRNN':
        args.current_processed_dataset_path = min_dfscode_tensor_path

    if args.produce_graphs:
        mkdir(args.current_dataset_path)

        if args.graph_type in ['Lung', 'Breast', 'Leukemia', 'Yeast', 'All']:
            count = produce_graphs_from_raw_format(
                input_path, args.current_dataset_path, args.num_graphs,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)

        elif args.graph_type in ['ENZYMES']:
            count = produce_graphs_from_graphrnn_format(
                base_path, args.graph_type, args.current_dataset_path,
                num_graphs=args.num_graphs, node_invariants=node_invariants,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)

        elif args.graph_type in ['cora', 'citeseer']:
            count = produce_random_walk_sampled_graphs(
                base_path, args.graph_type, args.current_dataset_path,
                num_graphs=args.num_graphs, iterations=random_walk_iterations,
                num_factor=num_factor, min_num_nodes=min_num_nodes,
                max_num_nodes=max_num_nodes, min_num_edges=min_num_edges,
                max_num_edges=max_num_edges)
        
        elif args.graph_type in ['com_small', 'ego_small', 'grid', 'enz', 'planar', 'sbm']:
            count = produce_graphs_from_gdss_format(input_path, args.current_dataset_path)
        
        elif args.graph_type == 'lobster':
            count = produce_graphs_lobster(args.current_dataset_path)
        
        elif args.graph_type == 'ego':
            count = produce_graphs_ego(input_path, args.current_dataset_path)
            
        elif args.graph_type == 'proteins':
            count = produce_graphs_proteins(input_path, args.current_dataset_path)
            
        elif args.graph_type == 'point':
            count = produce_graphs_point(input_path, args.current_dataset_path)
        
        # elif args.graph_type in ['qm9', 'zinc']:
        #     count = produce_graphs_from_molecule(input_path, args.current_dataset_path, args.graph_type)
        elif args.graph_type == 'profold':
            count = produce_graphs_from_profold(input_path, args.current_dataset_path)
        print('Graphs produced', count)
    else:
        count = len([name for name in os.listdir(
            args.current_dataset_path) if name.endswith(".dat")])
        print('Graphs counted', count)

    # Produce feature map
    feature_map = mapping(args.current_dataset_path,
                          args.current_dataset_path + 'map.dict')
    print(feature_map)

    if args.note == 'DFScodeRNN' and args.produce_min_dfscodes:
        # Empty the directory
        mkdir(args.min_dfscode_path)

        start = time.time()
        graphs_to_min_dfscodes(args.current_dataset_path,
                               args.min_dfscode_path, args.current_temp_path)

        end = time.time()
        print('Time taken to make dfscodes = {:.3f}s'.format(end - start))

    if args.note == 'DFScodeRNN' and args.produce_min_dfscode_tensors:
        # Empty the directory
        mkdir(min_dfscode_tensor_path)

        start = time.time()
        min_dfscodes_to_tensors(args.min_dfscode_path,
                                min_dfscode_tensor_path, feature_map)

        end = time.time()
        print('Time taken to make dfscode tensors= {:.3f}s'.format(
            end - start))

    graphs = [i for i in range(count)]
    return graphs
