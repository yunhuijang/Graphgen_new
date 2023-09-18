import random
import time
import pickle
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import torch
import os
import json
import argparse
from datetime import datetime

from args import Args
from utils import create_dirs
from datasets.process_dataset import create_graphs
from datasets.preprocess import calc_max_prev_node, dfscodes_weights
from baselines.dgmg.data import DGMG_Dataset_from_file
from baselines.graph_rnn.data import Graph_Adj_Matrix_from_file
from graphgen.data import Graph_DFS_code_from_file
from model import create_model
from train import train



class model():
    def __init__(self, hparams):
        super().__init__(hparams)
    
    def start(args):
        create_dirs(args)

        random.seed(123)

        graphs = create_graphs(args)

        if args.graph_type in ['planar', 'sbm', 'proteins']:
            test_len = int(round(len(graphs)*0.2))
            train_len = int(round((len(graphs) - test_len)*0.8))
            val_len = len(graphs) - train_len - test_len
            graphs_train, val, graphs_validate = random_split(graphs, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(1234))

        elif args.graph_type in ['com_small', 'ego_small', 'grid', 'enz', 'ego']:
            train_size = 0.7
            val_size = 0.1
            test_size = 0.2
            seed = 52
            train_val, graphs_validate = train_test_split(graphs, train_size=train_size + val_size, shuffle=False)
            graphs_train, val = train_test_split(train_val, train_size=train_size / (train_size + val_size), random_state=seed, shuffle=True)
        elif args.graph_type in ['qm9', 'zinc']:
            train_size = 0.7
            val_size = 0.1
            test_size = 0.2
            with open(os.path.join('datasets', f'{args.graph_type}/valid_idx_{args.graph_type}.json')) as f:
                test_idx = json.load(f)
            if args.graph_type == 'qm9':
                test_idx = test_idx['valid_idxs']
            test_idx = [int(i) for i in test_idx]
            train_idx = [i for i in range(len(graphs)) if i not in test_idx]
            graphs_validate = [graphs[i] for i in test_idx]
            train_val = [graphs[i] for i in train_idx]
            graphs_train, _ = train_test_split(train_val, train_size=train_size / (train_size + val_size), random_state=42, shuffle=True)
        elif args.graph_type in ['point', 'lobster']:
            val_size = 0.2
            train_size = 0.7
            graphs_train = graphs[int(val_size*len(graphs)):int((train_size+val_size)*len(graphs))]
            val = graphs[:int(val_size*len(graphs))]
            graphs_validate = graphs[int((train_size+val_size)*len(graphs)):]
        else:
            # random.shuffle(graphs)
            graphs_train = graphs[: int(0.70 * len(graphs))]
            graphs_validate = graphs[int(0.70 * len(graphs)): int(0.90 * len(graphs))]

        # show graphs statistics
        print('Model:', args.note)
        print('Device:', args.device)
        print('Graph type:', args.graph_type)
        print('Training set: {}, Validation set: {}'.format(
            len(graphs_train), len(graphs_validate)))

        # Loading the feature map
        with open(args.current_dataset_path + 'map.dict', 'rb') as f:
            feature_map = pickle.load(f)

        print('Max number of nodes: {}'.format(feature_map['max_nodes']))
        print('Max number of edges: {}'.format(feature_map['max_edges']))
        print('Min number of nodes: {}'.format(feature_map['min_nodes']))
        print('Min number of edges: {}'.format(feature_map['min_edges']))
        print('Max degree of a node: {}'.format(feature_map['max_degree']))
        print('No. of node labels: {}'.format(len(feature_map['node_forward'])))
        print('No. of edge labels: {}'.format(len(feature_map['edge_forward'])))

        # Setting max_prev_node / max_tail_node and max_head_node
        if args.note == 'DFScodeRNN' and args.weights:
            feature_map = {
                **feature_map,
                **dfscodes_weights(args.min_dfscode_path, graphs_train, feature_map, args.device)
            }

        dataset_train = Graph_DFS_code_from_file(
            args, graphs_train, feature_map)
        dataset_validate = Graph_DFS_code_from_file(
            args, graphs_validate, feature_map)


        dataloader_train = DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.num_workers)
        dataloader_validate = DataLoader(
            dataset_validate, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)

        model = create_model(args, feature_map)

        train(args, dataloader_train, model, feature_map, dataloader_validate)

        
    def add_args(parser):
        
        time = '{0:%Y-%m-%d_%H:%M:%S}'.format(datetime.now())
        
        parser.add_argument("--time", type=str, default=time)
        
        parser.add_argument("--graph_type", type=str, default="planar")
        parser.add_argument("--note", type=str, default='DFScodeRNN')
        parser.add_argument("--device", type=str, default='cuda')
        
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--hidden_size_dfscode_rnn", type=int, default=256)
        parser.add_argument("--embedding_size_dfscode_rnn", type=int, default=92)
        parser.add_argument("--embedding_size_timestamp_output", type=int, default=512)
        parser.add_argument("--embedding_size_vertex_output", type=int, default=512)
        parser.add_argument("--embedding_size_edge_output", type=int, default=512)
        parser.add_argument("--dfscode_rnn_dropout", type=float, default=0.2)       
        parser.add_argument("--loss_type", type=str, default="BCE")       
        parser.add_argument("--weights", action="store_true")  
        
        parser.add_argument("--current_min_dfscode_path", type=str, default=None)
        parser.add_argument("--current_processed_dataset_path", type=str, default=None)
        
        parser.add_argument("--rnn_type", type=str, default='LSTM')
        parser.add_argument("--num_layers", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--lr", type=float, default=0.003)
        parser.add_argument("--gamma", type=float, default=0.003)
        parser.add_argument("--milestones", type=list, default=[100, 200, 400, 800])
        parser.add_argument("--gradient_clipping", action="store_false")
        
        parser.add_argument("--model_save_path", type=str, default='model_save/')
        parser.add_argument("--dataset_path", type=str, default='datasets/')
        parser.add_argument("--current_dataset_path", type=str, default=f'datasets/{parser.parse_args().graph_type}/graphs/')
        parser.add_argument("--tensorboard_path", type=str, default='tensorboard/')
        
        parser.add_argument("--temp_path", type=str, default='tmp/')
        
        parser.add_argument("--save_model", action="store_false")
        parser.add_argument("--epochs_save", type=int, default=100)
        parser.add_argument("--epochs_validate", type=int, default=100)
        fname = parser.parse_args().note + '_' + parser.parse_args().graph_type
        parser.add_argument("--fname", type=str, default=fname)
        cur_tmp_path = parser.parse_args().temp_path + fname + '_' + time
        parser.add_argument("--current_temp_path", type=str, default=cur_tmp_path)
        cur_model_save_path = parser.parse_args().model_save_path + '_' + fname + '_' + time + '/'
        parser.add_argument("--current_model_save_path", type=str, default=cur_model_save_path)
        
        parser.add_argument("--clean_tensorboard", action="store_true")
        parser.add_argument("--clean_temp", action="store_true")
        parser.add_argument("--load_model", action="store_true")
        parser.add_argument("--log_tensorboard", action="store_true")
        
        parser.add_argument("--produce_graphs", action="store_false")
        parser.add_argument("--produce_min_dfscodes", action="store_false")
        parser.add_argument("--produce_min_dfscode_tensors", action="store_false")
        
        
        
        return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model.add_args(parser)
    args = parser.parse_args()
    # args = args.update_args()
    model.start(args)