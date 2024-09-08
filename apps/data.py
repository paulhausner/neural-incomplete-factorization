from glob import glob

import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch_geometric.data import Data

from torch_geometric.loader import DataLoader


def matrix_to_graph_sparse(A, b):
    edge_index = torch.tensor(list(map(lambda x: [x[0], x[1]], zip(A.row, A.col))), dtype=torch.long)
    edge_features = torch.tensor(list(map(lambda x: [x], A.data)), dtype=torch.float)
    node_features = torch.tensor(list(map(lambda x: [x], b)), dtype=torch.float)

    # diag_elements = edge_index[:, 0] == edge_index[:, 1]
    # node_features = edge_features[diag_elements]
    # node_features = torch.cat((node_features, torch.tensor(list(map(lambda x: [x], b)), dtype=torch.float)), dim=1)
    
    # Embed the information into data object
    data = Data(x=node_features, edge_index=edge_index.t().contiguous(), edge_attr=edge_features)
    return data


def matrix_to_graph(A, b):
    return matrix_to_graph_sparse(coo_matrix(A), b)


def graph_to_matrix(data, normalize=False):
    A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr[:, 0].squeeze(), requires_grad=False)
    b = data.x[:, 0].squeeze()
    
    if normalize:
        b = b / torch.linalg.norm(b)
    
    return A, b


def get_dataloader(dataset, n=0, batch_size=1, spd=True, mode="train", size=None, graph=True):
    # Setup datasets
    
    if dataset == "random":
        data = FolderDataset(f"./data/Random/{mode}/", n, size=size, graph=graph)
    
    else:
        raise NotImplementedError("Dataset not implemented, Available: random")
    
    # Data Loaders
    if mode == "train":
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=1, shuffle=False)
    
    return dataloader


class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, n, graph=True, size=None) -> None:
        super().__init__()
        
        self.graph = True
        assert self.graph, "Graph keyword is depracated, only graph=True is supported."
                
        if n != 0:
            if self.graph:
                self.files = list(filter(lambda x: x.split("/")[-1].split('_')[0] == str(n), glob(folder+'*.pt')))
            else:
                self.files = list(filter(lambda x: x.split("/")[-1].split('_')[0] == str(n), glob(folder+'*.npz')))
        else:
            file_ending = "pt" if self.graph else "npz"
            self.files = list(glob(folder+f'*.{file_ending}'))
        
        if size is not None:
            assert len(self.files) >= size, f"Only {len(self.files)} files found in {folder} with n={n}"
            self.files = self.files[:size]
        
        if len(self.files) == 0:
            raise FileNotFoundError(f"No files found in {folder} with n={n}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if self.graph:
            g = torch.load(self.files[idx], weights_only=False)
        
        else:
            # deprecated...
            d = np.load(self.files[idx], allow_pickle=True)
            g = matrix_to_graph(d["A"], d["b"])
        
        return g
