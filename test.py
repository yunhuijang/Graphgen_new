
import pickle

with open('/home/yhjang/graph_compression_generation/graphgen/graphs/DFScodeRNN_ego_small_2023-09-18_17:02:41/50/graph0.dat', 'rb') as f:
    graph = pickle.load(f)

print(graph)