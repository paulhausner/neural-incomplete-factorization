import os

import numpy as np
import torch
import scipy
from scipy.sparse import coo_matrix

from data import matrix_to_graph


def generate_sparse_random(n, alpha=1e-4, random_state=0, sol=False, ood=False):
    # We add to spd matricies since the sparsity is only enforced on the cholesky decomposition
    # generare a lower trinagular matrix
    # Random state
    rng = np.random.RandomState(random_state)
    
    if alpha is None:
        alpha = rng.uniform(1e-4, 1e-2)
    
    # this is 1% sparsity for n = 10 000
    sparsity = 10e-4
    
    # create out of distribution samples
    if ood:
        factor = rng.uniform(0.22, 2.2)
        sparsity = factor * sparsity
    
    nnz = int(sparsity * n ** 2)
    rows = [rng.randint(0, n) for _ in range(nnz)]
    cols = [rng.randint(0, n) for _ in range(nnz)]
    
    uniques = set(zip(rows, cols))
    rows, cols = zip(*uniques)
    
    # generate values
    vals = np.array([rng.normal(0, 1) for _ in cols])
    
    M = coo_matrix((vals, (rows, cols)), shape=(n, n))
    I = scipy.sparse.identity(n)
    
    # create spd matrix
    A = (M @ M.T) + alpha * I
    print(f"Generated matrix with {100 * (A.nnz / n**2) :.2f}% non-zero elements: ({A.nnz} non-zeros)")
    
    # right hand side is uniform
    b = rng.uniform(0, 1, size=n)
    
    # We want a high-accuracy solution, so we use a direct sparse solver here.
    # only produce when in test mode
    if sol:
        # generate solution using dense method for accuracy reasons
        x, _ = scipy.sparse.linalg.cg(A, b)
        
    else:
        x = None
    
    return A, x, b


def create_dataset(n, samples, alpha=1e-2, graph=True, rs=0, mode='train', solution=False):
    if mode != 'train':
        assert rs != 0, 'rs must be set for test and val to avoid overlap'
    
    print(f"Generating {samples} samples for the {mode} dataset.")
    
    for sam in range(samples):
        # generate solution only for val and test
        
        A, x, b = generate_sparse_random(n, random_state=(rs + sam), alpha=alpha, sol=solution,
                                         ood=(mode=="test_ood"))
        
        if graph:
            graph = matrix_to_graph(A, b)
            if x is not None:
                graph.s = torch.tensor(x, dtype=torch.float)
            graph.n = n
            torch.save(graph, f'./data/Random/{mode}/{n}_{sam}.pt')
        else:
            A = coo_matrix(A)
            np.savez(f'./data/Random/{mode}/{n}_{sam}.npz', A=A, b=b, x=x)


if __name__ == '__main__':
    # create the folders and subfolders where the data is stored
    os.makedirs(f'./data/Random/train', exist_ok=True)
    os.makedirs(f'./data/Random/val', exist_ok=True)
    os.makedirs(f'./data/Random/test', exist_ok=True)
    
    # create 10k dataset
    n = 10_000
    alpha=10e-4
    
    create_dataset(n, 1000, alpha=alpha, mode='train', rs=0, graph=True, solution=True)
    create_dataset(n, 10, alpha=alpha, mode='val', rs=10000, graph=True)
    create_dataset(n, 100, alpha=alpha, mode='test', rs=103600, graph=True)
