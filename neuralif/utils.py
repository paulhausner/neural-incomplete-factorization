import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
import scipy
import scipy.sparse

import time
import os
import psutil

from torch_geometric.utils import coalesce, remove_self_loops, to_torch_coo_tensor, to_edge_index


class TwoHop(torch_geometric.transforms.BaseTransform):
    
    def forward(self, data):
        assert data.edge_index is not None
        edge_index, edge_attr = data.edge_index, data.edge_attr
        num_nodes = data.num_nodes

        adj = to_torch_coo_tensor(edge_index, size=num_nodes)

        adj = adj @ adj

        edge_index2, _ = to_edge_index(adj)
        edge_index2, _ = remove_self_loops(edge_index2)

        edge_index = torch.cat([edge_index, edge_index2], dim=1)

        if edge_attr is not None:
            # We treat newly added edge features as "zero-features":
            edge_attr2 = edge_attr.new_zeros(edge_index2.size(1),
                                             *edge_attr.size()[1:])
            edge_attr = torch.cat([edge_attr, edge_attr2], dim=0)

        data.edge_index, data.edge_attr = coalesce(edge_index, edge_attr,
                                                   num_nodes)

        return data


def gradient_clipping(model, clip=None):
    # track the gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
            
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
    return total_norm


def save_dict_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        file.write(json.dumps(dictionary))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def num_non_zeros(P):
    return torch.linalg.norm(P.flatten(), ord=0)


def frob_norm_sparse(data):
    return torch.pow(torch.sum(torch.pow(data, 2)), 0.5)
        

def filter_small_values(A, threshold=1e-5):
    # only keep the values above threshold
    return torch.where(torch.abs(A) < threshold, torch.zeros_like(A), A)


def plot_graph(data):
    # transofrm to networkx
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    # remove the self loops for readability
    filtered_edges = list(filter(lambda x: x[0] != x[1], g.edges()))
    nx.draw(g, edgelist=filtered_edges)
    plt.show()


def print_graph_statistics(data):
    print(data.validate())
    print(data.is_directed())
    print(data.num_nodes) 


def elapsed_since(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))


def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
 
 
def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()
        print("{}: memory before: {:,}, after: {:,}, consumed: {:,}; exec time: {}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before,
            elapsed_time))
        return result
    return wrapper


def test_spd(A):
    # the matrix should be symmetric positive definite
    np.testing.assert_allclose(A, A.T, atol=1e-6)
    assert np.linalg.eigh(A)[0].min() > 0


def kA_bound(cond, k):
    return 2 * ((torch.sqrt(cond) - 1) / (torch.sqrt(cond) + 1)) ** k


def eigenval_distribution(P, A):
    if P == None:
        return torch.linalg.eigh(A)[0]
    else:
        return torch.linalg.eigh(P@A@P.T)[0]


def condition_number(P, A, invert=False, split=True):
    if invert:
        if split:
            P = torch.linalg.solve_triangular(P, torch.eye(P.size()[0], device=P.device, requires_grad=False), upper=False)
        else:
            P = torch.linalg.inv(P)
    
    if split:
        # P.T@A@P is wrong!
        # Not sure what the difference is between P@A@P.T and P.T@P@A?
        return torch.linalg.cond(P@A@P.T)
    else:
        return torch.linalg.cond(P@A)


def l1_output_norm(P):
    # normalized output norm
    return torch.sum(torch.abs(P)) / P.size()[0]


def rademacher(n, m=1, device=None):
    if device == None:
        return torch.sign(torch.randn(n, m, requires_grad=False))
    else:
        return torch.sign(torch.randn(n, m, device=device, requires_grad=False))


def torch_sparse_to_scipy(A):
    A = A.coalesce()
    d = A.values().squeeze().numpy()
    i, j = A.indices().numpy()
    A_s = scipy.sparse.coo_matrix((d, (i, j)))
    
    return A_s


def gershgorin_norm(A, graph=False):
    
    if graph:
        row, col = A.edge_index
        agg = torch_geometric.nn.aggr.SumAggregation()
        
        row_sum = agg(torch.abs(A.edge_attr), row)
        col_sum = agg(torch.abs(A.edge_attr), col)
        
    else:
        # compute the normalization factor
        n = A.size()[0]
        
        # compute row and column sums
        row_sum = torch.sum(torch.abs(A.to_dense()), dim=1)
        col_sum = torch.sum(torch.abs(A.to_dense()), dim=0)
        
    gamma = torch.min(torch.max(row_sum), torch.max(col_sum))
    return gamma


time_function = lambda: time.perf_counter()
